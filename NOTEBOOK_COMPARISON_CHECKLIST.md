# Complete Notebook Comparison Checklist
## Local vs Kaggle Version Verification

**Date:** January 2026  
**Notebook:** `01_eda.ipynb`  
**Total Cells (Local):** 91

---

## Complete Section Structure

### Section 1: Environment & Dataset Inspection
- [ ] **1.1** Kaggle Environment & Input Validation
- [ ] Key elements: `INPUT_DIR`, `os.path`, environment setup

### Section 2: Spark Session Initialization
- [ ] **2.1** Spark Session Setup
- [ ] Key elements: `SparkSession`, `builder`, `getOrCreate()`

### Section 3: Data Loading & Structural Validation
- [ ] **3.1** Load Training Dataset (Spark)
- [ ] Key elements: `spark.read.csv`, `fraudTrain.csv`

### Section 4: Data Quality & Target Overview
- [ ] **4.1** Fraud Statistics
- [ ] **4.2** Missing Values Analysis
- [ ] Key elements: `is_fraud`, missing values analysis, summary stats

### Section 5: Key Findings from Initial Exploration
- [ ] Dataset size and structure summary

### Section 6: Timezone Resolution & Temporal Feature Engineering
- [ ] **6.1** Constants and Utilities
  - [ ] `GRID_SIZE = 0.5` constant
  - [ ] `resolve_timezone_with_grid()` helper function
- [ ] **6.2** Load and Prepare ZIP Reference Table
  - [ ] **CRITICAL:** `.groupBy('lat_grid', 'lng_grid').agg(first('timezone'))`
  - [ ] **CRITICAL:** Should NOT have `zip_ref` in select (prevents row explosion)
  - [ ] Key elements: `zip_ref_df`, `lat_grid`, `lng_grid`
- [ ] **6.3** Convert UTC Timestamps to Local Time
  - [ ] **6.3.3** Production Fallback
    - [ ] **CRITICAL:** Single-pass aggregation for validation
    - [ ] **CRITICAL:** Achieves 100% coverage for both `merchant_local_time` and `customer_local_time`
  - [ ] Key elements: `from_utc_timestamp`, `merchant_local_time`, `customer_local_time`

### Section 7: Temporal Fraud Pattern Analysis
- [ ] **7.0** Helper Functions for Temporal Analysis
  - [ ] **CRITICAL:** `validate_temporal_coverage()` function
  - [ ] **CRITICAL:** `aggregate_fraud_by_dimension()` function
- [ ] **7.1** Fraud Patterns by Hour
  - [ ] **7.1.1** Data Validation
  - [ ] **7.1.2** Hourly Fraud Aggregation
  - [ ] **7.1.3** Hourly Fraud Visualizations
  - [ ] **7.1.4** Key Findings Summary
- [ ] **7.2** Fraud Patterns by Day of Week
  - [ ] **7.2.1** Data Validation
  - [ ] **7.2.2** Day of Week Aggregation
  - [ ] **7.2.3** Weekend vs Weekday Comparison
    - [ ] **CRITICAL FIX:** Defensive checks present:
      - [ ] `if 'daily_analysis_df' not in locals():` check
      - [ ] `if "day_of_week" not in daily_analysis_df.columns:` check
      - [ ] `if "is_weekend" not in daily_analysis_df.columns:` check
  - [ ] **7.2.4** Key Findings Summary
- [ ] **7.3** Fraud Patterns by Month (Seasonal Analysis)
  - [ ] Month extraction and aggregation
- [ ] **7.4** Weekend vs Weekday Deep Dive
  - [ ] Should be minimal (references Section 7.2.3)
  - [ ] Code checks if `weekend_stats` exists
- [ ] **7.5** Time Bin Analysis
  - [ ] **CRITICAL FIX:** Defensive check present:
    - [ ] `if "hour" not in train_df.columns:` check
    - [ ] Creates `hour` from `merchant_local_time` if missing
- [ ] **7.6** Temporal Analysis Summary & Conclusions
  - [ ] Comprehensive summary with findings
  - [ ] Feature enhancement recommendations
  - [ ] Production recommendations

---

## Critical Code Elements to Verify

### Section 6.2 - ZIP Reference Table (Row Explosion Fix)
**Must have:**
```python
zip_ref_df = (
    ...
    .select("lat_grid", "lng_grid", "timezone")  # NO zip_ref here
    .distinct()
    .groupBy("lat_grid", "lng_grid")
    .agg(first("timezone").alias("timezone"))
    .cache()
)
```

**Must NOT have:**
- `zip_ref` in the `.select()` statement (causes row explosion)

### Section 6.3.3 - Production Fallback (100% Coverage)
**Must have:**
```python
# Single-pass aggregation
counts_before = train_df.agg(
    count("*").alias("total"),
    spark_sum(when(col("merchant_local_time").isNotNull(), 1).otherwise(0)).alias("merchant_valid"),
    spark_sum(when(col("customer_local_time").isNotNull(), 1).otherwise(0)).alias("customer_valid")
).collect()[0]

# Fallback logic
train_df = train_df.withColumn(
    "merchant_local_time",
    when(col("merchant_local_time").isNull(), col("customer_local_time"))
    .otherwise(col("merchant_local_time"))
)
# ... (additional fallback levels)
```

**Result:** 100% coverage for both time columns

### Section 7.2.3 - Weekend Analysis (is_weekend Fix)
**Must have:**
```python
# Ensure daily_analysis_df exists and has required columns (defensive check)
if 'daily_analysis_df' not in locals():
    daily_analysis_df, coverage, valid, total = validate_temporal_coverage(...)

# Ensure day_of_week column exists
if "day_of_week" not in daily_analysis_df.columns:
    daily_analysis_df = daily_analysis_df.withColumn("day_of_week", ...)

# Ensure is_weekend column exists (defensive check)
if "is_weekend" not in daily_analysis_df.columns:
    daily_analysis_df = daily_analysis_df.withColumn(
        "is_weekend",
        when(col("day_of_week").isin([1, 7]), 1).otherwise(0)
    )
```

### Section 7.5 - Time Bin Analysis (hour Fix)
**Must have:**
```python
# Ensure hour column exists (defensive check)
if "hour" not in train_df.columns:
    if "merchant_local_time" not in train_df.columns:
        raise ValueError("merchant_local_time column is required...")
    train_df = train_df.withColumn(
        "hour",
        hour(col("merchant_local_time"))
    )
```

### Section 7.0 - Helper Functions
**Must have:**
```python
def validate_temporal_coverage(df, time_column, analysis_name):
    """Validate temporal data coverage before analysis."""
    # ... implementation

def aggregate_fraud_by_dimension(df, dimension_col, dimension_name, cache_name=None):
    """Aggregate fraud statistics by a single dimension."""
    # ... implementation
```

---

## Comparison Results

After checking Kaggle, note any differences:

### Sections Missing in Kaggle:
- [ ] Section 7.0 (Helper Functions)
- [ ] Section 7.6 (Summary)
- [ ] Other: _______________

### Critical Fixes Missing in Kaggle:
- [ ] Section 6.2 - Row explosion fix (groupBy)
- [ ] Section 6.3.3 - Single-pass aggregation
- [ ] Section 7.2.3 - is_weekend defensive checks
- [ ] Section 7.5 - hour defensive check
- [ ] Other: _______________

### Code Differences Found:
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

---

## Action Items

After comparison:
- [ ] Update local notebook to match Kaggle (if Kaggle has working version)
- [ ] Update Kaggle notebook with fixes from local (if local has fixes)
- [ ] Sync both versions to be identical
- [ ] Verify all sections run successfully in both environments

---

**Last Updated:** Based on local notebook analysis  
**Total Sections:** 7 main sections, 25+ subsections  
**Critical Fixes:** 5 major fixes to verify
