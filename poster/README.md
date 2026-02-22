# Poster Pipeline README

This folder contains everything needed to generate and manage ISEF poster assets.

## Directory Layout

```text
poster/
  tools/                # Poster generation scripts
  output/               # Generated outputs (all artifacts)
    figures/            # PNG/SVG/CSV figure assets
    model_summary.csv   # Model performance summary table
    FIGURE_INDEX.md     # Figure inventory + model summary
    ISEF_Poster_Draft.pptx
```

## Tools

### Data source (all scripts)

All poster scripts in this folder use cached local neuron exports instead of live Allen API calls:

- `scripts/cell_types/ephys_features.csv`
- `scripts/cell_types/cells.json`

The same files are shared by:

- full poster asset generation
- CV/overfitting explainer
- concept figure (`decision_tree_random_forest_concept`)

### 1) `poster/tools/generate_isef_assets.py`
Generates the full poster asset set from local cached data:

- dataset overview plots
- PCA and feature diagnostics
- node sweep (Decision Tree + Random Forest) CSV/PNG
- confusion matrices
- ROC/PR curves (binary tasks)
- feature importance plots
- thumbnail contact sheets
- `model_summary.csv`
- `FIGURE_INDEX.md`
- `ISEF_Poster_Draft.pptx`

Default input data:
- `scripts/cell_types/ephys_features.csv`
- `scripts/cell_types/cells.json`

Default output:
- `poster/output`

### 2) `poster/tools/make_cv_overfitting_explainer.py`
Generates a methods explainer figure showing:

- CV fold train/test split diagram
- Decision Tree train/test accuracy vs node count
- Random Forest train/test accuracy vs node count
- overfitting region annotation

Defaults:
- input figure dir: `poster/output/figures`
- output file: `poster/output/figures/cv_split_overfitting_explainer.png`
- y-axis range: `0.6` to `1.0`

### 3) `poster/tools/make_tree_forest_concept_figure.py`
Generates a conceptual figure grounded in the local neuron dataset:

- real learned decision tree for `dendrite_type`
- random forest voting on an example neuron
- top random-forest feature importances
- summary text with dataset/model metrics

Default output:
- `poster/output/figures/decision_tree_random_forest_concept.png`

### 4) `poster/run_all_tools.sh`
Runs the full poster pipeline end-to-end in order:

- `generate_isef_assets.py`
- `make_cv_overfitting_explainer.py`
- `make_tree_forest_concept_figure.py`

Default command:

```bash
bash poster/run_all_tools.sh
```

## Dataset snapshot (LnCellData / cell-types cache)

Current local snapshot provenance:

- Source files: `scripts/cell_types/ephys_features.csv` and `scripts/cell_types/cells.json`.
- These two files are generated from the Allen SDK cell-types cache manifest in `scripts/cell_types/manifest.json`.

What is in the snapshot (from the current files checked into this workspace):

- `cells.json`: 2,333 specimens.
- `ephys_features.csv`: 2,333 rows (one row per specimen).
- 1-to-1 merge on `specimen_id`: 2,333 matched rows.
- Dendrite labels present: `spiny` 1,213, `aspiny` 1,000, `sparsely spiny` 120.
- Species represented in snapshot: 2 (`Homo Sapiens`, `Mus musculus`).
- Brain areas represented in snapshot: 30 unique area acronyms.
- Analysis subset used by `decision_tree_random_forest_concept` and most posters: `spiny + aspiny` only.
- `spiny + aspiny` rows used downstream: 2,213 (1,213 spiny + 1,000 aspiny).
- Live API check confirms total specimen rows for this query: 2,333 (as of the latest API call).
- For perspective, Allen API `Specimen` totals are much larger (265,937) because that model includes all specimen records beyond the Cell Types subset.

Important accuracy note:

- These counts verify internal consistency of the local snapshot.
- They do not prove completeness of the entire Allen Cell Types catalog unless you compare against a fresh live source query.
- If you want the “full” dataset claim, rerun source extraction from the remote API and regenerate `scripts/cell_types/*` so the snapshot timestamp is explicit.

Why this is not necessarily the full Allen Cell Types DB:

- The snapshot is a **cache snapshot**, not a live query.
- The local files use API shape from `ApiCellTypesSpecimenDetail` and can drift as the Allen API updates.
- The dataset query used for these files excludes cases outside the current success criteria used by the cache extraction.
- Some label classes are present but not modeled (for example `sparsely spiny` is present in `cells.json` but excluded from default downstream targets).

Quick local and API consistency check:

```bash
python3 poster/tools/verify_allen_celltypes_api.py
```

The script calls the Allen API with `num_rows=0` only, so it returns metadata counts without downloading full feature matrices.

Interpretation:

- If local counts diverge from API totals, your cache snapshot is stale or built using a different filter.
- If you only want a live coverage check, run `verify_allen_celltypes_api.py` before regenerating poster inputs.

## How To Run

Run from repository root (`/Users/krha/code/cell_classification`).

```bash
source .venv/bin/activate
```

### Full pipeline

```bash
python poster/tools/generate_isef_assets.py
```

### CV overfitting explainer only

```bash
python poster/tools/make_cv_overfitting_explainer.py
```

Optional custom call:

```bash
python poster/tools/make_cv_overfitting_explainer.py \
  --input-csv node_sweep_dendrite_type_electrophysiology.csv \
  --output poster/output/figures/cv_split_overfitting_explainer_electrophysiology.png \
  --y-min 0.6 \
  --y-max 1.0
```

### Concept figure only

```bash
python poster/tools/make_tree_forest_concept_figure.py
```

### Concept figure custom call

```bash
python poster/tools/make_tree_forest_concept_figure.py \
  --data-dir scripts/cell_types \
  --output poster/output/figures/decision_tree_random_forest_concept.png
```
```

## Outputs (What You Get)

Top-level outputs in `poster/output`:

- `ISEF_Poster_Draft.pptx`
- `model_summary.csv`
- `FIGURE_INDEX.md`
- `figures/` (all plots and intermediate CSVs)

`model_summary.csv` is regenerated by `generate_isef_assets.py` and tracks per-task/per-feature
model metrics that are also used by the explanation figures.

Main figure files in `poster/output/figures`:

- `dataset_overview.png`
- `species_dendrite_heatmap.png`
- `pca_embeddings.png`
- `feature_quality_stats.png`
- `feature_correlation_heatmap.png`
- `decision_tree_performance_grid.png`
- `confusion_matrix_grid.png`
- `cv_split_overfitting_explainer*.png` and `*.svg`
- `decision_tree_random_forest_concept.png` and `.svg`

Generated families (patterns):

- `node_sweep_<task>_<feature>.csv`
- `node_sweep_<task>_<feature>.png`
- `confusion_<task>_<feature>.png`
- `feature_importance_<task>_<feature>.png`
- `roc_pr_<task>_<feature>.png`

where:
- `<task>` is one of: `dendrite_type`, `brain_area`, `species_dendrite`
- `<feature>` is one of: `electrophysiology`, `morphology`, `all_features`

Thumbnail files:

- `figures/morphology_thumbnail_sheet.png`
- `figures/ephys_thumbnail_sheet.png`
- `figures/thumbs/` (downloaded raw thumbnails used in contact sheets)

## Quick Verification

```bash
find poster/output -maxdepth 2 -type f | sort
```
