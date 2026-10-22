# Smart Buildings Probability Project

Team project beginning with Task 1: dataset understanding, preparation, and Chapter 2 probability applications. The same dataset and notebook foundation are intended to carry forward into later course tasks.

## Project files

### Data

- `buildings_200_features.csv` - 200 accepted generated buildings, 54 columns.
- `rejected_seeds.csv` - 82 generator seeds rejected during the run.
- `feature_catalog.csv` - definitions for the 54 CSV columns and 79 simulation-only observation features.

### Task 1 work

- `M1_data_foundation.ipynb` - Member 1's Colab-ready notebook foundation, including data checks, minimal preprocessing, the rejected-seed table, and shared event columns.
- `TEAM_WORKFLOW_AND_NOTEBOOK_STUDY_GUIDE.md` - member responsibilities, GitHub branch workflow, and a walkthrough of the Member 1 notebook.
- `Task_01.pdf` - assignment brief.
- `Task1_Team_Guide.pdf` - team responsibilities and workflow.
- `smart_buildings_200_report (1) (1).pdf` - dataset generation and validation report.

## Run the notebook

Open `M1_data_foundation.ipynb` in Google Colab. Put the three CSV files in the shared Drive folder `stats_project`, or upload them when prompted. Run **Runtime > Restart session and run all**. The notebook preserves raw data and creates an analysis copy; expected preprocessing is 200 rows x 54 columns to 200 rows x 37 columns after dropping 17 constant columns.

The shared event columns are `A_large`, `B_extra_ahu`, `F2`, `F3`, `F4`, and `O2`. Teammates should use these names in their sections.

## Collaboration

Keep one master notebook. Develop new sections in separate copies or branches, then merge them into the master when ready. Do not edit the master concurrently. Keep the source CSV files unchanged and make data transformations reproducible in notebook code.

## Data scope

These rows are generated building configurations, not a random sample of real buildings. Building-level probabilities are conditional on the generator accepting a seed. The files contain no energy-use or comfort outcome.
