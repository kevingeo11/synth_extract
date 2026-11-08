- **conditions:** measurement conditions stated in the paper for this measurement. Each condition has:
  - `name`: [[CONDITION_NAMES_LONG]].
  - `type`: `exact`, `mean_sd`, `range`, `approximate`, `lower_bound`, or `upper_bound`.
  - `value`: a number when numeric (e.g., [[COND_NUMBER_EXAMPLES]]), otherwise text (e.g., [[COND_TEXT_EXAMPLES]]).
  - `unit`: e.g., [[COND_UNIT_EXAMPLES]]; `null` for non-numeric values.
  - `description`: `null`, or a short note (required when `name` is `other`).

  Include only conditions the paper states. Use an empty list if none.
