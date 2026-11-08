"""Property extraction agents for one property in one paper.

Two prompt variants, one class each:

* :class:`PaperPropertyExtractor` (prompt 2, ``per_paper``): one request per
  batch of target samples. The model reports every value of the property in
  the paper and links each value to a listed sample with a ``match_basis``.
* :class:`SamplePropertyExtractor` (prompt 3, ``with_context``): one request
  per sample. The model reports the values of one target sample; the paper's
  other samples are given only to tell the samples apart.

Each instance serves one property. The classes define the prompts, the
structured-output contract and the clean-up of one paper's responses (batch
filtering, de-duplication and flags). Loading papers, persistence, retries and
the overall scheduling are owned by the caller, as for ``SampleExtractor``.
Records are flagged rather than dropped, so filtering policy stays with the
caller; :meth:`PaperPropertyResult.to_properties` maps a result to the stored
:class:`Property` records.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from synth_extract.agents.llm import (
    LLMBackend,
    StructuredAgent,
    StructuredFailure,
    StructuredOutcome,
    StructuredSuccess,
    StructuredTask,
)

from .property_config import PROPERTY_CONFIG, STRONG_MATCHES
from .property_prompts import (
    DEFAULT_PROPERTY_PROMPT_DIR,
    fill_user_prompt,
    load_templates,
    render_system_prompt,
    render_user_template,
)
from .property_schemas import (
    FLAG_NAME_ONLY,
    RECORD_FIELDS,
    ExtractedRecord,
    PaperContext,
    PaperPropertyList,
    PaperPropertyRequest,
    PaperPropertyResult,
    RequestFailure,
    SamplePropertyList,
    SamplePropertyRequest,
    check_record,
    make_paper_response_format,
    make_response_format,
)


DEFAULT_TARGET_BATCH_SIZE = 15
DEFAULT_MAX_CONCURRENCY = 4


class _PropertyExtractorBase:
    """Prompt loading, hashing and configuration shared by both agents."""

    mode: ClassVar[str]

    def __init__(
        self,
        backend: LLMBackend,
        property: str,
        prompt_dir: str | Path = DEFAULT_PROPERTY_PROMPT_DIR,
    ) -> None:
        if property not in PROPERTY_CONFIG:
            raise ValueError(f"property must be one of {list(PROPERTY_CONFIG)}, got {property!r}")
        self.backend = backend
        self.property = property
        self.prompt_dir = Path(prompt_dir).expanduser().resolve()
        self._agent = StructuredAgent(backend)
        self.reload_prompts()

    def reload_prompts(self) -> None:
        """Reload the templates from disk and render them for this property."""
        templates = load_templates(self.mode, self.prompt_dir)
        self._system_prompt = render_system_prompt(self.property, templates)
        self._user_template = render_user_template(self.property, templates)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_template(self) -> str:
        return self._user_template

    def prompt_hash(self) -> str:
        """Hash of the rendered prompts, so config and template changes both count."""
        text = f"{self._system_prompt}\n\n{self._user_template}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def llm_config(self) -> dict[str, Any]:
        """Return non-secret backend and prompt configuration."""
        return {
            **self.backend.config(),
            "property": self.property,
            "mode": self.mode,
            "prompt_dir": str(self.prompt_dir),
            "prompt_hash": self.prompt_hash(),
        }

    def health_check(self) -> bool:
        self.backend.list_models()
        return True

    # ---- helpers for one paper -------------------------------------------------------------

    @staticmethod
    async def _gather(run_one, requests: Sequence[Any], semaphore: asyncio.Semaphore | None,
                      max_concurrency: int) -> list[StructuredOutcome[Any]]:
        limiter = semaphore or asyncio.Semaphore(max_concurrency)

        async def limited(request):
            async with limiter:
                return await run_one(request)

        results = await asyncio.gather(*(limited(r) for r in requests), return_exceptions=True)
        return [
            r if not isinstance(r, BaseException)
            else StructuredFailure(error_type="unknown", message=f"{type(r).__name__}: {r}")
            for r in results
        ]

    def _new_result(self, paper: PaperContext) -> PaperPropertyResult:
        return PaperPropertyResult(
            paper_uid=paper.paper_uid, property=self.property, mode=self.mode,
            prompt_hash=self.prompt_hash(),
        )

    @staticmethod
    def _add_usage(result: PaperPropertyResult, outcome: StructuredSuccess[Any]) -> None:
        usage = outcome.metadata.usage
        if usage is not None:
            result.prompt_tokens += usage.prompt_tokens or 0
            result.completion_tokens += usage.completion_tokens or 0


    @staticmethod
    def _drop_duplicates(result: PaperPropertyResult) -> None:
        """Remove exact duplicates of a record within one sample.

        Two linked records are duplicates when they belong to the same sample and
        every stored field is equal (``RECORD_FIELDS``, including the conditions
        and the description). Of each group the copy with the fewest flags is
        kept, at the position of the first copy. Records of different samples
        are never compared. Unlinked records are left as they are.
        """

        def number(value: Any) -> Any:
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value

        def key(item: ExtractedRecord) -> tuple[Any, ...]:
            record = item.record
            conditions = tuple(
                (c.name, c.type, number(c.value), c.unit, c.description) for c in record.conditions
            )
            fields = tuple(
                conditions if field == "conditions" else number(getattr(record, field))
                for field in RECORD_FIELDS
            )
            return (item.sample_id, *fields)

        kept: list[ExtractedRecord] = []
        position: dict[tuple[Any, ...], int] = {}
        for item in result.records:
            if not item.linked:
                kept.append(item)
                continue
            k = key(item)
            if k not in position:
                position[k] = len(kept)
                kept.append(item)
                continue
            result.duplicates_dropped += 1
            if len(item.flags) < len(kept[position[k]].flags):
                kept[position[k]] = item
        result.records = kept


class SamplePropertyExtractor(_PropertyExtractorBase):
    """Prompt 3: extract one property for one sample, with the other samples as context."""

    mode: ClassVar[str] = "with_context"

    def reload_prompts(self) -> None:
        super().reload_prompts()
        # Unsubscripted call: subscripting a slotted frozen dataclass fails on Python 3.10.
        self._task: StructuredTask[SamplePropertyRequest, SamplePropertyList] = StructuredTask(
            name=f"property_{self.property}_with_context",
            prompt_builder=self._build_user_prompt,
            output_model=SamplePropertyList,
            response_format=self.response_format(),
            system_prompt=self._system_prompt,
        )

    def response_format(self) -> dict[str, Any]:
        """Return the provider-facing strict JSON schema."""
        return make_response_format(self.property)

    def _build_user_prompt(self, request: SamplePropertyRequest) -> str:
        return fill_user_prompt(self._user_template, self.mode, {
            "paper_uid": request.paper_uid,
            "sample_id": str(request.target.sample_id),
            "sample_record": request.target.record_json(),
            "other_samples": "\n".join(s.list_json() for s in request.other_samples) or "[none]",
            "markdown_full_text": request.markdown_full_text,
        })

    # ---- one request -----------------------------------------------------------------------

    def extract(self, request: SamplePropertyRequest) -> StructuredOutcome[SamplePropertyList]:
        """Synchronously extract the property for one sample."""
        return self._agent.run(self._task, request)

    async def aextract(self, request: SamplePropertyRequest) -> StructuredOutcome[SamplePropertyList]:
        """Asynchronously extract the property for one sample."""
        return await self._agent.arun(self._task, request)

    def render_request(self, request: SamplePropertyRequest) -> str:
        """Render the exact backend request without calling the model."""
        return self._agent.render_request(self._task, request)

    # ---- one paper -------------------------------------------------------------------------

    @staticmethod
    def build_requests(paper: PaperContext, sample_ids: Sequence[int] | None = None) -> list[SamplePropertyRequest]:
        """One request per sample (all samples, or the given ones), in paper order."""
        wanted = None if sample_ids is None else set(sample_ids)
        if wanted is not None:
            unknown = wanted - {s.sample_id for s in paper.samples}
            if unknown:
                raise ValueError(f"sample_ids not in paper {paper.paper_uid}: {sorted(unknown)}")
        return [
            SamplePropertyRequest(
                paper_uid=paper.paper_uid,
                target=target,
                other_samples=[s for s in paper.samples if s.sample_id != target.sample_id],
                markdown_full_text=paper.markdown_full_text,
            )
            for target in paper.samples
            if wanted is None or target.sample_id in wanted
        ]

    def collect(self, paper: PaperContext, requests: Sequence[SamplePropertyRequest],
                outcomes: Sequence[StructuredOutcome[SamplePropertyList]]) -> PaperPropertyResult:
        """Turn one paper's responses into records with flags."""
        result = self._new_result(paper)
        result.requests = len(requests)
        for request, outcome in zip(requests, outcomes):
            sample_id = request.target.sample_id
            if isinstance(outcome, StructuredFailure):
                result.failures.append(RequestFailure(
                    sample_ids=[sample_id], error_type=outcome.error_type,
                    message=outcome.message, raw_response=outcome.raw_response,
                ))
                continue
            self._add_usage(result, outcome)
            for record in outcome.result.properties:
                result.records.append(ExtractedRecord(
                    sample_id=sample_id, record=record, flags=check_record(record, self.property),
                ))
        self._drop_duplicates(result)
        return result

    async def aextract_paper(
        self,
        paper: PaperContext,
        *,
        sample_ids: Sequence[int] | None = None,
        semaphore: asyncio.Semaphore | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> PaperPropertyResult:
        """Run one request per sample concurrently and collect the records.

        Pass a shared ``semaphore`` to limit requests across papers; otherwise
        at most ``max_concurrency`` requests of this paper run at once.
        """
        requests = self.build_requests(paper, sample_ids)
        outcomes = await self._gather(self.aextract, requests, semaphore, max_concurrency)
        return self.collect(paper, requests, outcomes)

    def extract_paper(self, paper: PaperContext, *, sample_ids: Sequence[int] | None = None) -> PaperPropertyResult:
        """Synchronous version of :meth:`aextract_paper` (one request at a time)."""
        requests = self.build_requests(paper, sample_ids)
        return self.collect(paper, requests, [self.extract(r) for r in requests])


class PaperPropertyExtractor(_PropertyExtractorBase):
    """Prompt 2: extract one property for a whole paper and link values to samples."""

    mode: ClassVar[str] = "per_paper"

    def _task(self, request: PaperPropertyRequest) -> StructuredTask[PaperPropertyRequest, PaperPropertyList]:
        # The schema limits sample_id to this paper's samples, so it is built per request.
        return StructuredTask(
            name=f"property_{self.property}_per_paper",
            prompt_builder=self._build_user_prompt,
            output_model=PaperPropertyList,
            response_format=self.response_format([s.sample_id for s in request.samples]),
            system_prompt=self._system_prompt,
        )

    def response_format(self, sample_ids: Sequence[int]) -> dict[str, Any]:
        """Return the provider-facing strict JSON schema for these sample IDs."""
        return make_paper_response_format(self.property, list(sample_ids))

    def _build_user_prompt(self, request: PaperPropertyRequest) -> str:
        targets = ("all samples in the list" if request.all_targets
                   else ", ".join(map(str, request.target_sample_ids)))
        return fill_user_prompt(self._user_template, self.mode, {
            "paper_uid": request.paper_uid,
            "target_sample_ids": targets,
            "sample_list": "\n".join(s.list_json() for s in request.samples),
            "markdown_full_text": request.markdown_full_text,
        })

    # ---- one request -----------------------------------------------------------------------

    def extract(self, request: PaperPropertyRequest) -> StructuredOutcome[PaperPropertyList]:
        """Synchronously extract the property for one batch of target samples."""
        return self._agent.run(self._task(request), request)

    async def aextract(self, request: PaperPropertyRequest) -> StructuredOutcome[PaperPropertyList]:
        """Asynchronously extract the property for one batch of target samples."""
        return await self._agent.arun(self._task(request), request)

    def render_request(self, request: PaperPropertyRequest) -> str:
        """Render the exact backend request without calling the model."""
        return self._agent.render_request(self._task(request), request)

    # ---- one paper -------------------------------------------------------------------------

    @staticmethod
    def build_requests(paper: PaperContext, batch_size: int = DEFAULT_TARGET_BATCH_SIZE) -> list[PaperPropertyRequest]:
        """Split the samples into target batches; each request lists every sample."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        ids = [s.sample_id for s in paper.samples]
        return [
            PaperPropertyRequest(
                paper_uid=paper.paper_uid,
                samples=paper.samples,
                target_sample_ids=ids[i:i + batch_size],
                markdown_full_text=paper.markdown_full_text,
            )
            for i in range(0, len(ids), batch_size)
        ]

    def collect(self, paper: PaperContext, requests: Sequence[PaperPropertyRequest],
                outcomes: Sequence[StructuredOutcome[PaperPropertyList]]) -> PaperPropertyResult:
        """Turn one paper's responses into records with flags.

        A link counts only from the request that targeted that sample. A value
        with ``match_basis: none`` or no ``sample_id`` is unlinked and kept once,
        even if several batches return it; it does not become a Property.
        A ``name_only`` link gets the ``name_only`` flag; ``same_entry`` and
        ``same_description`` links get no link flag.
        """
        result = self._new_result(paper)
        result.requests = len(requests)
        seen_unlinked: set[tuple[str, str, str]] = set()
        for request, outcome in zip(requests, outcomes):
            targets = set(request.target_sample_ids)
            if isinstance(outcome, StructuredFailure):
                result.failures.append(RequestFailure(
                    sample_ids=list(request.target_sample_ids), error_type=outcome.error_type,
                    message=outcome.message, raw_response=outcome.raw_response,
                ))
                continue
            self._add_usage(result, outcome)
            for record in outcome.result.properties:
                sample_id = None if record.match_basis == "none" else record.sample_id
                if sample_id is not None and sample_id not in targets:
                    continue  # belongs to another batch's targets
                if sample_id is None:
                    key = (record.paper_label, record.value_raw, record.method)
                    if key in seen_unlinked:
                        continue
                    seen_unlinked.add(key)
                flags = check_record(record, self.property)
                if sample_id is not None and record.match_basis not in STRONG_MATCHES:
                    flags.append(FLAG_NAME_ONLY)
                result.records.append(ExtractedRecord(sample_id=sample_id, record=record, flags=flags))
        self._drop_duplicates(result)
        return result

    async def aextract_paper(
        self,
        paper: PaperContext,
        *,
        batch_size: int = DEFAULT_TARGET_BATCH_SIZE,
        semaphore: asyncio.Semaphore | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> PaperPropertyResult:
        """Run the paper's batch requests (usually one) and collect the records."""
        requests = self.build_requests(paper, batch_size)
        outcomes = await self._gather(self.aextract, requests, semaphore, max_concurrency)
        return self.collect(paper, requests, outcomes)

    def extract_paper(self, paper: PaperContext, *, batch_size: int = DEFAULT_TARGET_BATCH_SIZE) -> PaperPropertyResult:
        """Synchronous version of :meth:`aextract_paper` (one request at a time)."""
        requests = self.build_requests(paper, batch_size)
        return self.collect(paper, requests, [self.extract(r) for r in requests])


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_TARGET_BATCH_SIZE",
    "PaperPropertyExtractor",
    "SamplePropertyExtractor",
]
