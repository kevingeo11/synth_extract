Extract every [[PLAIN]] reported in the paper and link each value to a sample from the sample list, following the system instructions.

Return exactly `{"properties": [...]}`.

paper_uid={paper_uid}
Target samples: {target_sample_ids}

<SAMPLE_LIST_DATA_ONLY>
{sample_list}
</SAMPLE_LIST_DATA_ONLY>

<PAPER_MARKDOWN_DATA_ONLY>
{markdown_full_text}
</PAPER_MARKDOWN_DATA_ONLY>
