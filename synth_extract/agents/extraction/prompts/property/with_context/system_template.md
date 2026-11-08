# Task

Extract every **[[LABEL]]** reported for **one requested polymer/material sample** from the supplied paper.

Use the target record and the list of the paper's other samples **only to identify the target sample**. All values and measurement details must come from the paper, including its text, tables, captions, footnotes, and supplementary text.

Return exactly:

`{"properties": [...]}`

If the paper reports no [[SHORT]] for this sample, return `{"properties": []}`.

## 1. Which values belong to the sample

Extract a [[SHORT]] only if the paper clearly reports it for **this exact sample**.

- Do not extract values for other samples, precursors, components, or reference materials.
- Do not extract values quoted from other publications.
- Do not extract trends without a number (e.g., [[TREND_EXAMPLE]]).
- Do not extract [[EXCLUDE]].
- Sample names can be shared by several samples, so identify the target from its whole record, especially the preparation or characterization conditions it describes, and extract a value only when the paper reports it for a sample made under those same conditions.
- The paper's other samples are listed only so you can tell the samples apart. Compare the target record with them, and extract a value only if the paper reports it for the target and not for one of the other listed samples or for a material that is not in the list. Never extract values for the other samples.
- If it is unclear whether a value belongs to this sample, skip it.

One record per distinct reported value:

- The same value given in both text and table → one record.
- The same [[SHORT]] measured with [[SEPARATE_RECORDS]] → separate records.

## 2. Fields

- **property:** always `[[PROPERTY]]`.
- **method:** the technique used to measure this [[SHORT]]: [[METHOD_LIST]]. General instrument descriptions in the Methods section apply to this sample's measurement.
[[SUBTYPE_FIELD]]
- **value_raw:** the value exactly as written, including ±, ranges, inequalities, and unit (e.g., [[VALUE_RAW_EXAMPLES]]).
- **value_type** and **value:** classify the reported form and set a single number:

[[VALUE_TABLE]]

[[VALUE_NOTE]]

  Use `not_observed` only when the paper explicitly states that no [[SHORT]] was detected for this sample.

- **unit:** [[UNIT_HELP]]
[[CONDITIONS_FIELD]]
- **origin:** `measured`, or `calculated` if the value comes from a model or equation (e.g., [[CALC_EXAMPLES]]).
[[DESCRIPTION_FIELD]]

## 3. Source fidelity

Never invent, infer, calculate, convert, or average values or conditions. Keep each value with its correct method and conditions. If a detail is not stated, use `unspecified`, `null`, or an empty list.

## 4. Output

Before returning, verify that:

- Every record is for the requested sample.
- No cited, plotted-only, or other-sample values are included.
- `value` follows the table above.
- `subtype` matches `method`.
- `description` is fully specified and contains no value, sample name, or table or figure number.

Return **only** the JSON object. No explanations or extra fields.
