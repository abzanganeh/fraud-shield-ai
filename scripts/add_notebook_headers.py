"""
Add standardized header cells to notebooks 02-07 across colab, kaggle, local.
Inserts: Notebook Header, Project Information, Data Attribution (NB02), Problem Statement, Scope.
Adds condensed Project Overview before Setup.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_INFO = """| **Attribute** | **Details** |
| :--- | :--- |
| **Author** | Alireza Barzin Zanganeh |
| **Contact** | abarzinzanganeh@gmail.com |
| **Date** | January 18, 2026 |
| **Project Type** | Capstone Project |
| **Pipeline** | Multi-environment deployment: Kaggle, Colab, and Local notebooks; Python scripts (`scripts/`); Streamlit app (`app/`) |
| **Repository** | [GitHub - fraud-shield-ai](https://github.com/abzanganeh/fraud-shield-ai) |"""

DATA_ATTRIBUTION = """**Data Source Attribution:**
ZIP code and timezone data provided by [SimpleMaps US ZIP Code Database](https://simplemaps.com/data/us-zips). Free version used for EDA. Use in production requires linking back as per their license."""

PROBLEM_STATEMENT = """**Problem Statement**
The objective is to design and implement a comprehensive fraud detection system capable of identifying fraudulent transactions with high accuracy and scalability. The system will leverage supervised learning to identify known fraud patterns from labeled historical data, and deep learning to model complex relationships and sequential transaction behaviors. The solution will focus on minimizing false positives, maximizing fraud recall, and maintaining scalability for high-volume real-time transaction processing."""

PROJECT_OVERVIEW = """**Project Overview**
This project implements a comprehensive fraud detection system using multiple machine learning approaches. The pipeline is implemented in Kaggle, Colab, and Local notebook versions. See the repository for full experiment logs and key learnings."""

SCOPE_TEMPLATES = {
    2: """**Notebook Scope & Constraints**
This notebook handles **preprocessing** (Spark MLlib pipeline, train/val/test split, feature engineering). Explicitly out of scope: model training.""",
    3: """**Notebook Scope & Constraints**
This notebook handles **feature importance and selection** using RF, XGBoost, LightGBM, Mutual Information. Explicitly out of scope: model training.""",
    4: """**Notebook Scope & Constraints**
This notebook trains **supervised models** (XGBoost, Logistic Regression, Random Forest) with iterative feature addition. Explicitly out of scope: deep learning.""",
    5: """**Notebook Scope & Constraints**
This notebook develops **deep learning models** (MLP, ResNet, LSTM) with Focal Loss and class-weighted BCE. Explicitly out of scope: transformer architectures.""",
    6: """**Notebook Scope & Constraints**
This notebook develops **transformer models** (FT-Transformer, Self-Attention MLP) for tabular fraud detection. Explicitly out of scope: ensemble methods.""",
    7: """**Notebook Scope & Constraints**
This notebook implements **hybrid ensemble** combining XGBoost and MLP via soft voting, weighted ensemble, and stacking. Explicitly out of scope: single-model training.""",
}

NOTEBOOK_CONFIGS = {
    "colab": {
        2: ("02-local-preprocessing.ipynb", "Notebook 2", "Spark MLlib Preprocessing Pipeline", "Colab execution - run first cell to install dependencies."),
        3: ("03-local-feature-engineering.ipynb", "Notebook 3", "Feature Engineering & Selection", "Colab execution - run first cell to install dependencies."),
        4: ("04-local-supervised-models.ipynb", "Notebook 4", "Supervised Models (XGBoost, etc.)", "Colab execution - run first cell to install dependencies."),
        5: ("05-local-deep-learning.ipynb", "Notebook 5", "Deep Learning Models", "Colab execution - run first cell to install dependencies."),
        6: ("06-local-transformers.ipynb", "Notebook 6", "Transformer Models", "Colab execution - run first cell to install dependencies."),
        7: ("07-local-hybrid-framework.ipynb", "Notebook 7", "Hybrid Ensemble Framework", "Colab execution - run first cell to install dependencies."),
    },
    "kaggle": {
        2: ("02-preprocessing.ipynb", "Notebook 2", "Spark MLlib Preprocessing Pipeline", "Kaggle execution - uses /kaggle/input paths."),
        3: ("03-feature-engineering.ipynb", "Notebook 3", "Feature Engineering & Selection", "Kaggle execution - uses /kaggle/input paths."),
        4: ("04-supervised-models.ipynb", "Notebook 4", "Supervised Models (XGBoost, etc.)", "Kaggle execution - uses /kaggle/input paths."),
        5: ("05-deep-learning.ipynb", "Notebook 5", "Deep Learning Models", "Kaggle execution - uses /kaggle/input paths."),
        6: ("06-transformers.ipynb", "Notebook 6", "Transformer Models", "Kaggle execution - uses /kaggle/input paths."),
        7: ("07-hybrid-framework.ipynb", "Notebook 7", "Hybrid Ensemble Framework", "Kaggle execution - uses /kaggle/input paths."),
    },
    "local": {
        2: ("02-local-preprocessing.ipynb", "Notebook 2", "Spark MLlib Preprocessing Pipeline", "Local execution - conda env fraud-shield."),
        3: ("03-local-feature-engineering.ipynb", "Notebook 3", "Feature Engineering & Selection", "Local execution - conda env fraud-shield."),
        4: ("04-local-supervised-models.ipynb", "Notebook 4", "Supervised Models (XGBoost, etc.)", "Local execution - conda env fraud-shield."),
        5: ("05-local-deep-learning.ipynb", "Notebook 5", "Deep Learning Models", "Local execution - conda env fraud-shield."),
        6: ("06-local-transformers.ipynb", "Notebook 6", "Transformer Models", "Local execution - conda env fraud-shield."),
        7: ("07-local-hybrid-framework.ipynb", "Notebook 7", "Hybrid Ensemble Framework", "Local execution - conda env fraud-shield."),
    },
}


def build_header_cell(nb_num: int, nb_name: str, title: str, objective: str, env_note: str, include_data_attr: bool) -> dict:
    parts = [
        f"# **{title}** {objective}",
        "**Project:** Fraud Shield AI",
        f"**Notebook:** {nb_name}",
        f"**Objective:** {objective}",
        "",
        f"**Note:** {env_note}",
        "",
        "## Project Information",
        "",
        PROJECT_INFO,
        "",
    ]
    if include_data_attr:
        parts.extend([DATA_ATTRIBUTION, "", ""])
    parts.extend([PROBLEM_STATEMENT, "", SCOPE_TEMPLATES[nb_num]])
    text = "\n".join(parts)
    lines = text.split("\n")
    source = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build_overview_cell() -> dict:
    text = "## Project Overview\n\nThis project implements a comprehensive fraud detection system using multiple machine learning approaches. The pipeline is implemented in Kaggle, Colab, and Local notebook versions. See the repository for full experiment logs and key learnings."
    lines = text.split("\n")
    source = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def has_header(cells: list) -> bool:
    """Check if notebook already has the new header (Project Information with Pipeline)."""
    for cell in cells[:3]:
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", [])
            text = "".join(src) if isinstance(src, list) else src
            if "Pipeline** | Implemented in three environments" in text:
                return True
    return False


def add_headers_to_notebook(folder: str, nb_num: int):
    config = NOTEBOOK_CONFIGS[folder][nb_num]
    nb_name = config[0]
    title = config[1]
    env_note = config[3]
    path = PROJECT_ROOT / f"{folder}_notebooks" / nb_name
    if not path.exists():
        print(f"  Skip {path} (not found)")
        return
    with open(path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    if has_header(cells):
        print(f"  Skip {path} (already has header)")
        return
    objective = config[2]
    header_cell = build_header_cell(nb_num, nb_name, title, objective, env_note, include_data_attr=(nb_num == 2))
    overview_cell = build_overview_cell()
    new_cells = [header_cell, overview_cell] + cells
    nb["cells"] = new_cells
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Updated {path}")


def main():
    for folder in ["colab", "kaggle", "local"]:
        print(f"\n{folder}:")
        for nb_num in range(2, 8):
            add_headers_to_notebook(folder, nb_num)


if __name__ == "__main__":
    main()
