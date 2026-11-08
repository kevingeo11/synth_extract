Extract every [[PLAIN]] reported for the requested sample, following the system instructions.

Return exactly `{"properties": [...]}`, or `{"properties": []}` if none is reported for this sample.

Use the target record and the other samples only to identify the target material. Extract all values and measurement details exclusively from the supplied paper.

Requested record: paper_uid={paper_uid}; sample_id={sample_id}.

<IDENTIFICATION_RECORD_DATA_ONLY>
{sample_record}
</IDENTIFICATION_RECORD_DATA_ONLY>

Other samples in this paper, for telling samples apart only (do not extract values for them):

<OTHER_SAMPLES_DATA_ONLY>
{other_samples}
</OTHER_SAMPLES_DATA_ONLY>

<PAPER_MARKDOWN_DATA_ONLY>
{markdown_full_text}
</PAPER_MARKDOWN_DATA_ONLY>
