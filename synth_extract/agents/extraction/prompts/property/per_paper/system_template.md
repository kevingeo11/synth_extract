# Task

Extract every **[[LABEL]]** value reported in the supplied paper and link each value to a sample from the supplied sample list.

The sample list contains every sample recorded for this paper. Use it **only to identify samples**. All values and measurement details must come from the paper, including its text, tables, captions, footnotes, and supplementary text.

Return exactly:

`{"properties": [...]}`

If the paper reports no [[SHORT]] for the target samples, return `{"properties": []}`.

## 1. Which values to report

Report every [[SHORT]] value the paper gives for a target sample, and every [[SHORT]] value that matches no sample in the list. Skip values that clearly belong to a listed sample that is not a target.

- Do not report values quoted from other publications.
- Do not read values off plots. Use only numbers stated in text, tables, or captions.
- Do not report trends without a number.
- Do not report [[EXCLUDE]].

One record per distinct reported value:

- The same value given in both text and table → one record.
- The same [[SHORT]] measured with [[SEPARATE_RECORDS]] → separate records.

## 2. Linking a value to a sample

For each value, first record what the paper says about the material, then choose the sample.

- **paper_label:** the name or label the paper uses for the material next to the value (e.g., the table row label).
- **identifying_data:** short text with the data reported together with the value that identify the material: table entry, preparation conditions, composition, molecular weight, dispersity. Use `""` if none.
- **match_basis:** how the value is tied to the chosen sample:
  - `same_entry`: the paper refers the value to the same sample entry as the record. The paper's label for the material equals the record's `identifier` or one of its `aliases` or the paper places the value in the same table entry, run, or batch as the record. No other listed sample has that label or entry, and nothing reported with the value contradicts the record.
  - `same_description`: the preparation conditions or characterization data given with the value (e.g.,
    composition, feed ratio, molecular weight, dispersity, processing) agree with the record's `description`,
    and with no other listed sample. Numbers must agree as reported; close values belong to a different sample.
  - `name_only`: only the name matches; nothing else ties the value to this sample.
  - `none`: no listed sample matches, or the data given with the value contradict every candidate.
- **sample_id:** the `sample_id` of the matched sample, or `null` when `match_basis` is `none`.

Rules:

- Several samples can share a name. A shared name alone never decides between them; compare the identifying data with each record.
- If the identifying data disagree with a sample record (e.g., a different molecular weight or different preparation conditions), the value does not belong to that sample.
- If no listed sample agrees with the identifying data, use `match_basis: none` and `sample_id: null`. This is a correct and common answer.

## 3. Value fields

- **property:** always `[[PROPERTY]]`.
- **method:** the technique used to measure this [[SHORT]]: [[METHOD_LIST]]. Table footnotes and general instrument descriptions in the Methods section count as stated.
[[SUBTYPE_FIELD]]
- **value_raw:** the value exactly as written, including ±, ranges, inequalities, and unit (e.g., [[VALUE_RAW_EXAMPLES]]).
- **value_type** and **value:** classify the reported form and set a single number:

[[VALUE_TABLE]]

[[VALUE_NOTE]]

  Use `not_observed` only when the paper explicitly states that no [[SHORT]] was detected.

- **unit:** [[UNIT_HELP]]
[[CONDITIONS_FIELD]]
- **origin:** `measured`, or `calculated` if the value comes from a model or equation (e.g., [[CALC_EXAMPLES]]).
[[DESCRIPTION_FIELD]]

## 4. Source fidelity

Never invent, infer, calculate, convert, or average values or conditions. If a detail is not stated, use `unspecified`, `null`, `""`, or an empty list.

Before returning, verify that `subtype` matches `method` and that every `description` is fully specified and contains no value, sample name, or table or figure number.

Return **only** the JSON object. No explanations or extra fields.
