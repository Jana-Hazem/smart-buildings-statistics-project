# Member 1 slide draft

Use this content in the team deck. The guide assigns slides 1-3 to Member 1, while its detailed slide list assigns slide 2 (problem/objectives) to Member 5. Confirm the final slide ownership with the team; this draft covers the Member 1 dataset and preprocessing slides.

## Slide 1 - What is in the dataset?

**Headline:** 200 accepted smart-building configurations from a 282-seed generator run

- 3 CSV files: buildings (200 x 54), rejected seeds (82 x 2), feature catalog (133 x 4)
- One building row = one accepted generated configuration; `seed` links the attempted runs
- Fields cover metadata, configuration, derived building structure, and observation schema

**Visual:** a simple flow: 282 attempted seeds -> 200 accepted buildings + 82 rejected seeds. A small data-type strip can show numeric, categorical, text/identifier, and date/time fields. Note that 79 catalog entries describe simulation-only observations, not columns in the building CSV.

**Speaker note:** The building table is the analysis population for building-level probabilities. Rejected seeds are retained as a separate part of the generator-run sample space. The feature catalog has 133 definitions: the 54 stored columns plus 79 simulation-only observation features.

## Slide 3 - Preprocessing: preserve rows, simplify the analysis frame

**Headline:** 200 rows stay; 54 source columns become an analysis frame with constant fields removed

- Preserve all three raw tables; copy before transforming
- Standardize column names by dropping the group prefix; remove constant columns from `df` only
- No missing or duplicate building rows; keep valid large buildings (up to 305 zones)

**Visual:** `200 x 54 -> 200 x 37` with a callout: “0 rows removed; 17 constant columns omitted from analysis frame.” Add small check marks for seed/hash uniqueness and the zone/VAV and AHU arithmetic checks.

**Speaker note:** Removing constants does not change any event probability because a constant field cannot split this sample into meaningful event/non-event groups. All later building probabilities use the 200 accepted buildings, so they are conditional on the generator accepting the seed. Outliers remain because they are valid generator outputs; Member 4's sensitivity analysis checks whether results depend on the largest 5% by zone count.

## Slide 2 coordination note

The guide's role summary gives slides 1-3 to Member 1, but its detailed slide plan gives slide 2 (“Problem, objectives, why it matters”) to Member 5. To avoid duplicating content, let Member 5 own slide 2 unless the team agrees otherwise.
