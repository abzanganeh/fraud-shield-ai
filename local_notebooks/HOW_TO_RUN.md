# How to Run the Local Notebook

## Prerequisites Setup

### 1. Install Java 11 (Required for PySpark)

```bash
# Install Java 11
brew install openjdk@11

# Set JAVA_HOME (add to ~/.zshrc for persistence)
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 11)' >> ~/.zshrc
source ~/.zshrc

# Verify Java installation
java -version
```

### 2. Activate Conda Environment

The `fraud-shield` environment already exists. Activate it:

```bash
conda activate fraud-shield
```

### 3. Verify PySpark Works

```bash
python -c "from pyspark.sql import SparkSession; print('PySpark OK')"
```

### 4. Prepare Data Files

Place your data files in the `data/input/` directory:

```bash
# Copy your data files to the input directory
cp /path/to/your/fraudTrain.csv data/input/
cp /path/to/your/uszips.csv data/input/  # if needed

# Verify files are in place
ls -lh data/input/
```

**Required files:**
- `data/input/fraudTrain.csv` - Main training dataset (required)
- `data/input/uszips.csv` - ZIP code data (if needed for geographic features)

### 5. Directory Structure

The following directories should exist (already created):
```
fraud-shield-ai/
├── data/
│   ├── input/          # Place data files here
│   └── checkpoints/    # Checkpoints will be saved here
│       └── results/    # Analysis results will be saved here
└── local_notebooks/
    └── 01-local-fraud-detection-eda.ipynb
```

## Running the Notebook

### Option 1: Jupyter Notebook (Recommended)

```bash
# Navigate to project root
cd /Users/abzanganeh/Desktop/projects/fraud-shield-ai

# Activate conda environment
conda activate fraud-shield

# Start Jupyter Notebook
jupyter notebook

# Then navigate to: local_notebooks/01-local-fraud-detection-eda.ipynb
```

### Option 2: JupyterLab

```bash
# Activate conda environment
conda activate fraud-shield

# Start JupyterLab
jupyter lab

# Open: local_notebooks/01-local-fraud-detection-eda.ipynb
```

### Option 3: VS Code / Cursor

1. Open the notebook file: `local_notebooks/01-local-fraud-detection-eda.ipynb`
2. Select the `fraud-shield` kernel (top right of notebook)
3. Run cells sequentially

## Important Notes

1. **Working Directory**: Run the notebook from the project root (`fraud-shield-ai/`) so relative paths work correctly.

2. **Memory Configuration**: The notebook is configured for 32GB RAM with:
   - Spark driver memory: 12GB
   - Spark executor memory: 12GB
   - If you have less RAM, adjust these values in the Spark configuration cell.

3. **Checkpoints**: The notebook uses checkpoints to save progress. They will be saved in `data/checkpoints/` and persist between runs.

4. **First Run**: The first run will take longer as it processes the full dataset. Subsequent runs will use cached checkpoints.

## Troubleshooting

### Java Not Found
```bash
# Verify JAVA_HOME is set
echo $JAVA_HOME

# If empty, set it:
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
```

### PySpark Import Error
```bash
# Verify PySpark is installed in the conda environment
conda activate fraud-shield
python -c "import pyspark; print(pyspark.__version__)"
```

### Data File Not Found
- Ensure `fraudTrain.csv` is in `data/input/fraudTrain.csv`
- Check the path is relative to project root
- Verify file permissions: `ls -l data/input/fraudTrain.csv`

### Memory Issues
- Reduce Spark memory settings in the configuration cell if you have less than 32GB RAM
- Close other applications to free up memory

## Quick Start Checklist

- [ ] Java 11 installed and JAVA_HOME set
- [ ] Conda environment `fraud-shield` activated
- [ ] PySpark verified working
- [ ] Data files in `data/input/` directory
- [ ] Running from project root directory
- [ ] Jupyter/VS Code using `fraud-shield` kernel

Once all checkboxes are complete, you're ready to run the notebook!
