# Team Workflow and Notebook Study Guide

## Important note about the shared Colab link

The provided Colab URL currently opens a Google Drive page that says **“You need access”** for the signed-in account. This guide therefore explains the working Member 1 notebook in this repository, `M1_data_foundation.ipynb`. To explain a different notebook cell by cell, share the Colab file with the signed-in Google account or download the `.ipynb` file and add it to this repository.

## Repository and branches

Repository: [smart-buildings-statistics-project](https://github.com/Jana-Hazem/smart-buildings-statistics-project)

- `main` - the shared reference/master branch, currently containing the Member 1 foundation.
- `member-2` - Probability A: sample space, axioms, complement, counting, addition rule, and rejected-seed analysis.
- `member-3` - Probability B: conditional probability, multiplication, total probability, and Bayes' theorem.
- `member-4` - independence analysis, validation, Monte Carlo, bootstrap, and sensitivity.
- `member-5` - introduction, insights, limitations, charts, and presentation assembly.

Each member branch starts from the same current `main` commit. Branches isolate work; they do not grant permission to push. The repository is currently public, so anyone can read it. Only the owner and collaborators with write permission can push branches. The owner must invite teammates by their GitHub usernames under repository Settings > Collaborators, or teammates can fork the public repository and submit pull requests from their forks.

## How each member should work

### 1. Get the repository

Install Git or GitHub Desktop, then clone:

```bash
git clone https://github.com/Jana-Hazem/smart-buildings-statistics-project.git
cd smart-buildings-statistics-project
```

If the repo is already cloned, fetch the latest branch list:

```bash
git fetch origin
```

### 2. Switch to your own branch

Replace `member-2` below with your assigned branch name:

```bash
git switch member-2
git pull origin member-2
```

Do all work on that branch. Do not commit directly to `main`. Before changing the notebook, coordinate with Member 1 so two people do not edit the shared master at the same time.

### 3. Draft your contribution

The team guide asks each member to work in a personal notebook copy. Keep a draft such as `T1_M2_draft.ipynb` while developing. Use the same prepared DataFrame `df` and the same event-column names from the master. Do not rename the shared columns or change the raw data. For every application, use this order:

1. Question
2. Events and assumptions
3. Rule and why it fits
4. Calculation and result
5. Interpretation in the context of this generated dataset

Put the first three items in a Markdown/text cell before the code; put calculation results and interpretation after it. Use random seed `42` for simulation or resampling.

### 4. Save, commit, and push

For example, to add your draft notebook:

```bash
git add T1_M2_draft.ipynb
git commit -m "Draft Member 2 probability applications"
git push -u origin member-2
```

If you make more changes later:

```bash
git add T1_M2_draft.ipynb
git commit -m "Complete Member 2 applications"
git push
```

GitHub Desktop can do the same steps with its **Commit** and **Push origin** buttons.

### 5. Hand off for integration

Open a pull request from your branch to `main` when your draft is ready. In the PR description, list the notebook section numbers you completed and any variables your cells need. Member 1 reviews and integrates approved cells into `M1_data_foundation.ipynb`; Member 5 handles the final story and charts. Keep the draft separate until the integration is checked. Notebook files are JSON, so simultaneous edits to the same notebook can create merge conflicts even when people work on separate branches.

Before starting from a newer `main`, update your branch:

```bash
git fetch origin
git merge origin/main
```

If Git reports a conflict, stop and coordinate with Member 1 rather than choosing one notebook version blindly. Once Member 1 integrates the contribution, everyone should use the updated master on `main` as the source for later tasks.

## Who owns what

| Member | Notebook work | Slide work |
|---|---|---|
| 1 | Sections 1-4: load files, describe fields, quality checks, preprocessing, rejected-seed table, event contract; owns master merge | Dataset structure and preprocessing content (slides 1 and 3 in the detailed slide plan) |
| 2 | Sections 5.1-5.4: sample space, axioms/complement, counting, addition rule, rejected-seed probability | Slides 4-5 |
| 3 | Sections 5.5-5.8: conditional, multiplication, total probability, Bayes | Slides 6-7 |
| 4 | Section 5.9 and Section 6: independence, validation, Monte Carlo, bootstrap, sensitivity | Slides 8-9 |
| 5 | Section 0 introduction and Section 7 insights/limitations; charts and final QA | Slides 2 and 10-12; assembles deck |

The guide's role overview says Member 1 owns slides 1-3, but the detailed slide plan assigns slide 2 to Member 5. Confirm that split with the team before assembling the deck.

## Notebook walkthrough: `M1_data_foundation.ipynb`

### First Markdown cell: project context

This introduces the Smart Buildings dataset and states that probabilities describe generated buildings, not the real building stock. It also leaves a clear reminder for Member 5 to finalize the problem, objectives, and analysis roadmap. The assignment requires the notebook to start with a text cell, so do not move code above it.

### Section 1: setup and file loading

The code imports:

- `Path` to build file-system paths safely.
- `re` for extracting information from rejection-reason text.
- `numpy` for reproducible sampling and later probability work.
- `pandas` for tables and analysis.

It searches the notebook's current folder and the usual Colab Drive folder for the three CSVs. If they are not found and the notebook is running in Colab, it prompts for upload. It then loads three raw tables:

- `df_raw`: buildings, expected shape 200 x 54.
- `rej_raw`: rejected seeds, expected shape 82 x 2.
- `cat_raw`: feature catalog, expected shape 133 x 4.

The shape assertions stop the notebook early if a different or incomplete set of files was loaded. These raw DataFrames are preserved unchanged.

### Section 2: what the files and rows represent

The Markdown explains the link among the files. A building row is one accepted generated configuration. The accepted `seed` links it to the generator run; rejected seeds have a seed and text reason. The accepted and rejected seeds together form the full run of 282 consecutive seeds, 100000-100281. The feature catalog contains 133 definitions: the 54 columns stored in the CSV plus 79 simulation-only observation feature definitions.

### Sample code cell

`sample_cols` selects nine readable fields across metadata, configuration, derived structure, and observation schema. `df_raw.sample(5, random_state=42)` picks five records in a repeatable way. The fixed seed means another run shows the same sample, which makes review and screenshots consistent.

### Variable dictionary code cell

`classify_type()` assigns a high-level variable type from the column name and pandas dtype: date/time, numeric-discrete, numeric-continuous, text/identifier, or categorical.

`plain_meaning()` maps the feature names to plain-English labels. If a column has no hand-written label, it creates a readable fallback from the name. `catalog_csv` filters the feature catalog down to the 54 fields actually stored in the building CSV. The loop makes one dictionary row for each raw CSV column, including its group, meaning, type, and catalog definition. The assertion checks that no CSV column was missed.

The catalog is the source for group and technical definition. The plain-English meanings are explanations for readers; if a field needs a more precise interpretation, check the catalog and the source report.

### Section 3: minimal preprocessing

The data is already clean: the checks expect zero missing cells and zero duplicate building rows. Therefore no building rows are removed or repaired.

`df = df_raw.copy()` creates an analysis copy. Column names such as `derived__n_zones` become `n_zones`; only the first prefix is removed, so nested names such as `cfg__floor_plan.num_floors` become `floor_plan.num_floors`. The raw data stays unchanged.

The code detects columns with only one distinct value and drops them from `df` only. In this dataset there are 17 constants, so the analysis frame becomes 200 x 37. This makes the working frame smaller; it does not change event counts because constant columns cannot split rows into event and non-event groups. The full 54 columns remain available in `df_raw`.

The quality checks verify:

- no missing cells and no duplicate building rows;
- unique building seeds and unique array hashes;
- zone count equals VAV-box count in every row;
- total AHUs equal configured AHUs plus added AHUs;
- accepted and rejected seed sets do not overlap and together cover all 282 attempted seeds.

`preprocess_log` records row and column counts after each major step. Expected results are 200 x 54 -> 200 x 54 -> 200 x 37. The row count stays 200 throughout.

### Section 4: rejected-seed details and event columns

The rejected reason text contains phrases such as “produces 2 zones” and “smallest 7.92 m2.” The two regular expressions extract those values into `n_small_zones` and `smallest_m2`. `seeds` creates one row for each seed in 100000-100281 and labels it accepted when it appears in the building data.

The shared events on `df` are:

- `A_large`: `n_zones > 64`, where 64 is the median. The strict `>` means buildings tied at 64 are not in this event; there are 98 large buildings.
- `B_extra_ahu`: at least one AHU was added beyond the configured number; 146 buildings.
- `F2`, `F3`, `F4`: the building has 2, 3, or 4 floors; counts are 74, 63, and 63.
- `O2`: occupants per zone equals 2; 108 buildings.

These column names are the contract for Members 2-4. Use them exactly so the later notebook cells and validation code continue to work.

### Limitations paragraph

This section explains what results can and cannot mean. They describe generator output rather than real buildings. Building probabilities use the 200 accepted configurations, so they are conditional on generator acceptance. The 82 rejected seeds are useful for a separate seed-level question, but they are not building rows. The dataset has no energy or comfort outcome; observation vector lengths are derived from a formula; all buildings share one weather preset; some fields are redundant or constant; and 64 zones is an analytical threshold, not a physical definition of complexity.

### Sections 5-7: team placeholders

Section 5 is for probability applications by Members 2-4. Each application should state a question, events and assumptions, the rule and why it applies, the calculation, and an interpretation. Section 6 is for Member 4's validation and sensitivity analysis. Section 7 is for Member 5's findings, competition relevance, and final limitations. These headings are placeholders; they are not finished analyses yet.

## Key study questions

1. Why did we keep all 200 building rows? Because the input already has no missing or duplicate rows, and removing valid buildings would change the probability population.
2. What did preprocessing change? Names became easier to use and 17 constant columns were omitted from the analysis copy; the raw data and 200 observations remain unchanged.
3. Why are there two sample sizes? Building questions use 200 accepted buildings. Generator rejection questions use all 282 attempted seeds.
4. Why is `A_large` a strict greater-than comparison? The median is 64, and rows tied at 64 should remain in the not-large group.
5. Can we say large buildings cause extra AHUs? No. Both are related through the generator's design rules; this is generated-data association, not a causal study of real buildings.
