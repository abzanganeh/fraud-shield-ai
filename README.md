# Fraud Shield AI

Fraud detection pipeline: EDA, preprocessing, feature engineering, and modeling for transaction fraud. Uses PySpark for scale and sklearn/MLlib for preprocessing; timezone-aware features and checkpoints for reproducibility.

## Quick start

1. **Environment**
   ```bash
   conda env create -f environment.yml
   conda activate fraud-shield
   ```
2. **Java 11** (required for PySpark)
   ```bash
   brew install openjdk@11
   export JAVA_HOME=$(/usr/libexec/java_home -v 11)
   ```
3. **Data**  
   Place `fraudTrain.csv` and (if needed) `uszips.csv` in `data/input/`.
4. **Run**  
   From project root: open `local_notebooks/01-local-fraud-detection-eda.ipynb`, select kernel `fraud-shield`, run all. Then run `02-local-preprocessing.ipynb` (loads Section 8 checkpoint, engineers 30 leak-free features), then 04-07 for models.

## How to run (detail)

### Prerequisites

- Java 11, `JAVA_HOME` set
- Conda: `fraud-shield` env (see `environment.yml`)
- Data in `data/input/`: `fraudTrain.csv`, `uszips.csv` (for geo/timezone)

### Run options

- **Jupyter:** `jupyter notebook` or `jupyter lab` from project root; open `local_notebooks/01-local-fraud-detection-eda.ipynb`.
- **VS Code / Cursor:** Open the notebook, choose kernel `fraud-shield`, run cells in order.

Run notebooks in order: **01** (EDA + checkpoints) → **02** (preprocessing + feature engineering, 30 features) → 04–07 (models).

### Troubleshooting

- **Java:** `echo $JAVA_HOME`; if empty, `export JAVA_HOME=$(/usr/libexec/java_home -v 11)`.
- **PySpark:** `python -c "import pyspark; print(pyspark.__version__)"`.
- **LightGBM on macOS (libomp.dylib):** `brew install libomp` or `conda install -c conda-forge lightgbm`.
- **Data not found:** Ensure files are in `data/input/` and you run from project root.

## Project layout

- `local_notebooks/` – EDA, preprocessing, feature engineering, models (local run).
- `notebooks/` – Kaggle-oriented versions (paths differ).
- `scripts/` – Shared code (e.g. `timezone_pipeline.py`).
- `data/checkpoints/` – Parquet checkpoints from EDA/preprocessing (do not commit large data).
- `local_docs/` – Design doc, TODOs (local only; see `.gitignore`).

Design, checkpoints, and class choices are described in `local_docs/PROJECT_DESIGN_AND_APPROACH.md` (for first-time readers and handover).
