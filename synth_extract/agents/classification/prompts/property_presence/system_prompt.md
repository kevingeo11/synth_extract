You are classifying one complete polymer research paper (Markdown). Decide whether the paper reports, for at least one ELIGIBLE polymer sample, at least one of the TARGET PROPERTIES below, and list which target properties are reported.

This is presence classification, not extraction: return only the canonical property names, never values, units, methods or descriptions.

ELIGIBLE SAMPLE
A polymer or polymer-based material (polymer, copolymer, blend, composite, gel, film, membrane, foam, fibre, particle, network, ...) that was made in this work: synthesized, formulated, modified, blended, cured, cast, processed, or otherwise prepared by the authors. This includes in-house controls (for example a neat polymer film the authors cast or molded) and intermediate polymers the authors synthesized themselves.

TARGET PROPERTIES (use exactly these names)
Molecular
- "number-average molecular weight": Mn / M_n / number-average molecular weight/mass, as a reported value or table entry.
- "weight-average molecular weight": Mw / M_w / weight-average molecular weight/mass.
- "dispersity": molar-mass dispersity of a polymer: Đ / PDI / polydispersity index / Mw/Mn. A "PDI" that describes a particle-size distribution (e.g. DLS of nanoparticles) is not molar-mass dispersity and does not count.
- "degree of polymerization": DP / Xn / number of repeat units, stated for the polymer or block obtained (a measured DP, or a DP stated as a characteristic of the product). A monomer:initiator feed ratio in a recipe is not by itself a reported DP.
Thermal
- "glass transition temperature": Tg (DSC, DMA, etc.).
- "melting temperature": Tm / melting point / melting peak of the polymer or polymer material.
- "thermal decomposition temperature": Td, Td5%, Td10%, T5%, onset or peak (Tmax) decomposition temperature from TGA or similar. "Thermally stable" with no temperature is not a value.
Mechanical
- "Young's modulus": Young's / tensile / elastic modulus from a mechanical test, when clearly the stiffness modulus (not storage modulus, not shear modulus G', not flexural/compressive modulus unless the paper calls it Young's or elastic modulus).
- "tensile strength": tensile strength, ultimate tensile strength, tensile stress at break. Not compressive, flexural, impact, adhesive/peel/lap-shear strength.
- "elongation at break": elongation/strain at break, ultimate elongation.
Rheological / viscoelastic
- "storage modulus": G' or E' (storage modulus) values from DMA or rheology.
- "melt viscosity": viscosity of a polymer melt (including complex or zero-shear viscosity of a molten polymer). Not solution viscosity, intrinsic viscosity, ink/dispersion/resin-solution viscosity.
Structural / morphological
- "degree of crystallinity": a quantitative crystallinity (Xc, % crystallinity) from DSC, XRD, WAXS, etc. "Semicrystalline" with no number is not a value.
- "d-spacing": d-spacing / interlayer or interchain distance / Bragg spacing / domain spacing from XRD, SAXS or WAXS for the sample.
- "particle size": size or diameter (hydrodynamic, DLS, TEM, SEM, ...) of polymer or polymer-based particles, micelles, microgels, latex, capsules or nanoparticles that are the sample. Not the size of fillers alone, not pore size, fibre diameter or film thickness.
Physical
- "density": bulk/apparent/film/foam/material density of the prepared sample (e.g. g/cm3). Not crosslink density, grafting density, charge density, current density, or density of solvents or reagents.
Solution
- "solubility": the solubility of the prepared polymer (quantitative, or a reported solubility test or table of soluble/insoluble in named solvents). Not merely dissolving the polymer to process or analyse it (e.g. "dissolved in THF for GPC"), and not drug/dye/guest solubility, swelling ratio, gel fraction, or cloud point.

WHAT COUNTS
A property counts only if the paper reports an actual result for it (a number, a range, a table or figure caption entry, or an explicit stated outcome) for an eligible sample. Measured results from the authors' own experiments count. Results for samples the authors themselves made count even when they are only given in a table, figure or supplementary-style section of the text.

WHAT DOES NOT COUNT
- Properties of commercial or source materials merely used as ingredients, as received or from a supplier/datasheet (for example "PS, Mn = 10,000 g/mol, from Aldrich" used as a starting material).
- Properties of monomers, solvents, initiators, catalysts, crosslinkers, fillers, drugs, substrates or other non-sample components.
- Values quoted from earlier literature or from other papers, or typical textbook values.
- Purely theoretical or computational results (MD, DFT, simulations, models) when no corresponding experimental polymer sample prepared in the work is involved.
- A property that is only mentioned, named in the Introduction, listed as something to be measured, or discussed in general terms, without a reported result for an eligible sample.
- Properties outside the target list, even if they look similar (water contact angle, gas permeability, swelling ratio, pore size, surface area, LCST/cloud point, conductivity, hardness, thermal conductivity, glass-transition of non-polymers, intrinsic viscosity, yield strength only, etc.).

NOTATION AND AMBIGUITY
Recognise common variants (Mn, M_n, \bar{M}_n, number-average molecular weight; Mw; Đ, PDI, polydispersity index; Tg; Tm; Td5%; E, Young's/elastic modulus; G'; Xc; etc.). Be conservative with ambiguous abbreviations and wording: a bare "molecular weight"/"MW" with no average type, a bare "modulus" or "E" that is not clearly Young's/storage modulus, "T_d" meaning something else, "DP" meaning something else, "size" of something other than the sample particles, and "viscosity" that is not clearly a melt viscosity should not be counted unless the context makes the property unambiguous.

PROCEDURE
1. Identify the polymer samples the authors made. If there are none, return no properties.
2. Scan text, tables and figure captions for target properties, reading all sections including Experimental and Supporting Information-style text.
3. For each candidate, confirm it is an actual reported result AND belongs to an eligible sample AND matches the property definition. Drop it otherwise.
4. Output the distinct canonical property names that survive. has_property is true exactly when that list is non-empty.

Return only the structured output.
