Identify every distinct polymer material sample prepared in this paper and describe each one.

A sample is a polymer or polymer-based material made in this work. Samples are defined by synthesis: if the synthesis or formulation procedure differs (monomers, monomer ratio, composition, filler or additive loading, molecular weight, amounts, or reaction conditions such as temperature, time, catalyst, or solvent), it is a separate sample, even if the paper only reports it in a table or figure. Do not merge samples that differ in synthesis. When in doubt, list it as its own sample.

A chemical modification of a sample produces a new sample. Test or measurement conditions do not create new samples.

DO NOT INCLUDE AS SAMPLES

- commercial materials used as received, only as constituents of another sample;
- individual monomers, initiators, catalysts, solvents, crosslinkers, additives, fillers, drugs, or other components when they are not themselves the sample;
- characterization instruments or measurement substrates;
- computational models;
- materials reported only from other papers;
- intermediates that are only used to prepare another sample (their synthesis belongs to that sample).

DESCRIPTION

The description should contain enough information to understand and distinguish the sample without reading the surrounding paper. Include, when reported:

1. what the material is: polymer chemistry, composition, architecture, or form;
2. how it was made: the key synthesis steps and conditions needed to identify it;
3. what distinguishes its synthesis from the other samples in the paper;
4. explicit values when they distinguish samples ("PLLA block Mn 20 kg/mol", not "longer PLLA block").

Every statement must come from the paper. Do not infer, generalize, or add background knowledge. If something is not mentioned in the paper, it does not go in the description.

FIELDS

- `name`: the polymer or material name as written in the paper (for example, "PEG-b-PLLA" or "PVDF membrane"). It is not a code or label. Different samples may share a name.
- `identifier`: the label the paper uses for this specific sample (for example, "PEL-10", "Sample 7", or "MN3"); use an empty string if none is given.
- `aliases`: every other name, code, or identifier the paper uses for this same sample.
- `description`: the self-contained description specified above.

If no qualifying samples are present, return an empty `samples` array. Return only the structured output.
