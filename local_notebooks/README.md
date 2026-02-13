# Local Notebooks - Run Order

Run from project root with kernel `fraud-shield`. Execute in this order.

See [PLAN.md](PLAN.md) for rubric alignment and gap analysis.

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `01-local-fraud-detection-eda.ipynb` | EDA, timezone resolution, checkpoints |
| 2 | `02-local-preprocessing.ipynb` | Spark preprocessing, 37 features, train/val/test parquets |
| 3 | `04-local-supervised-models.ipynb` | LR, RF, XGBoost (Optuna-tuned) |
| 4 | `05-local-deep-learning.ipynb` | MLP, LSTM, ResNet |
| 5 | `06-local-transformers.ipynb` | Transformer model |
| 6 | `07-local-hybrid-framework.ipynb` | Ensemble + two-stage fraud detection |

**Optional:** `03-local-feature-engineering.ipynb` – feature importance analysis (not required for the pipeline).

## Two-Stage System (NB07)

- **Stage 1:** High-recall gate (flags ~0.5% of transactions)
- **Stage 2:** Auto-block (prob >= 0.90), Review queue (0.14–0.90), Cleared (< 0.14)
- Test recall: ~0.81; auto-block precision: ~0.82

## Prerequisites

- Java 11, `JAVA_HOME` set
- Conda env: `fraud-shield`
- Data: `fraudTrain.csv`, `fraudTest.csv`, `uszips.csv` in `data/input/`
