# Fraud Shield AI - Google Colab Notebooks

Run these notebooks on [Google Colab](https://colab.research.google.com/). Data is read from and written to **Google Drive**.

See [PLAN.md](PLAN.md) for rubric alignment and gap analysis.

## Setup

1. Create a folder in your Google Drive: `My Drive/fraud-shield-ai/`
2. Inside it, create: `data/input/`
3. Upload to `data/input/`:
   - `fraudTrain.csv`
   - `fraudTest.csv`
   - `uszips.csv` (for geo/timezone)

## Run Order

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | 01-local-fraud-detection-eda.ipynb | EDA, checkpoints |
| 2 | 02-local-preprocessing.ipynb | 37 features, train/val/test |
| 3 | 04-local-supervised-models.ipynb | XGBoost, LR, RF |
| 4 | 05-local-deep-learning.ipynb | MLP, LSTM, ResNet |
| 5 | 06-local-transformers.ipynb | Transformer |
| 6 | 07-local-hybrid-framework.ipynb | Ensemble + two-stage |

Optional: 03-local-feature-engineering.ipynb (feature importance).

## Usage

1. Upload a notebook to Colab (File > Upload notebook)
2. Run the first cell (installs PySpark and deps)
3. When prompted, authorize Google Drive mount
4. Run all cells in order

Outputs (parquets, models, results) are saved to `My Drive/fraud-shield-ai/`.
