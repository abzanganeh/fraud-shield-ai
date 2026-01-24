# Pending Fixes: Sections 7, 8, 9 & Analysis Roadmap

**Created:** January 24, 2026  
**Status:** Deferred for future implementation  
**Priority:** Medium (functionality works, but has bugs when checkpoints are missing)

---

## Overview

This document outlines three fixes identified during the comprehensive notebook review. These are **optional improvements** that address:
1. Analysis Roadmap mismatch with actual sections
2. Checkpoint save consistency (Sections 8 & 9)
3. Checkpoint load bug in Sections 7 & 8 (when checkpoint missing, `train_df` gets overwritten with `None`)

All fixes are **non-breaking** - the notebook works correctly when checkpoints exist. These fixes improve robustness when checkpoints are missing and the user wants to use in-memory `train_df`.

---

## 2.7: Analysis Roadmap (Section 1) Doesn't Match Actual Sections

### Issue

**Location:** Section 1, markdown cell with `## Analysis Roadmap`

**Current Content:**
```
1. Environment & Dataset Loading  
2. Data Quality Assessment  
3. Target Variable (Fraud) Overview  
4. Temporal Analysis (Timezone-Aware)  
5. Calendar-Level Patterns (Month, Seasonality)  
6. Key Insights & Next Steps
```

**Problem:** The actual notebook structure is:
- Section 1: Environment & Dataset Inspection
- Section 2: Spark Session Initialization
- Section 3: Data Loading & Structural Validation
- Section 4: Data Quality & Target Overview
- Section 5: Key Findings, Gaps & Amount Analysis
- Section 6: Timezone Resolution & Temporal Feature Engineering
- Section 7: Temporal Fraud Pattern Analysis
- Section 8: Geographical Analysis
- Section 9: Customer Demographics Analysis
- (Section 10: Credit Card Analysis - mentioned but not implemented)

### Fix

**Option A: Update Roadmap to Match Actual Structure**

Replace the Roadmap content with:

```markdown
## Analysis Roadmap

This notebook follows a strict, sequential EDA structure:

1. **Environment & Dataset Inspection** (Section 1)
   - Global imports, checkpoint configuration, Kaggle environment validation

2. **Spark Session Initialization** (Section 2)
   - Spark setup, checkpoint helper functions, ResultsManager

3. **Data Loading & Structural Validation** (Section 3)
   - Load training dataset, validate schema

4. **Data Quality & Target Overview** (Section 4)
   - Fraud statistics, missing values analysis

5. **Key Findings, Gaps & Amount Analysis** (Section 5)
   - Initial insights, amount distribution patterns

6. **Timezone Resolution & Temporal Feature Engineering** (Section 6)
   - Convert UTC timestamps to local time, create temporal features

7. **Temporal Fraud Pattern Analysis** (Section 7)
   - Hour, day of week, month, time bins, category analysis

8. **Geographical Analysis** (Section 8)
   - State, city population, distance, hotspots, merchant name

9. **Customer Demographics Analysis** (Section 9)
   - Age, gender, job/occupation patterns

Each section builds on validated assumptions from the previous one.
```

**Option B: Keep Generic Roadmap**

If you prefer a high-level roadmap, update to:

```markdown
## Analysis Roadmap

This notebook follows a sequential EDA structure (Sections 1-9):

1. **Setup & Data Loading** (Sections 1-3)
2. **Data Quality Assessment** (Section 4)
3. **Initial Analysis & Amount Patterns** (Section 5)
4. **Timezone Resolution** (Section 6)
5. **Temporal Patterns** (Section 7)
6. **Geographical Patterns** (Section 8)
7. **Demographics Patterns** (Section 9)

Each section builds on validated assumptions from the previous one.
```

### Implementation

**File:** `notebooks/01-fraud-detection-eda.ipynb`  
**Cell:** Markdown cell with `## Analysis Roadmap` (around line 50-60)

**Action:** Replace the roadmap content with Option A or B above.

---

## 2.8: Checkpoint Save Consistency (Sections 8 & 9)

### Issue

**Location:**
- Section 8.7: Save Section-Level Checkpoint code cell
- Section 9.4: Save Section-Level Checkpoint code cell

**Current Implementation:**
- Sections 5, 6, 7 use `checkpoint_manager.save_checkpoint()` with global `CHECKPOINT_SECTION*` paths
- Sections 8 and 9 **redefine** `CHECKPOINT_BASE_DIR` and `CHECKPOINT_SECTION*` locally
- Sections 8 and 9 use raw `train_df.write.mode("overwrite").parquet(...)` in `try/except` blocks

**Problem:**
- Inconsistent with Sections 5, 6, 7
- Redundant path definitions (already in Section 1.2)
- Manual error handling instead of using `CheckpointManager`'s built-in error handling

### Fix for Section 8.7

**Current Code:**
```python
# ============================================================
# SAVE SECTION-LEVEL CHECKPOINT: SECTION 8 (GEOGRAPHICAL FEATURES COMPLETE)
# ============================================================

# Use /kaggle/working explicitly for persistence across restarts
CHECKPOINT_BASE_DIR = "/kaggle/working/data/checkpoints"
CHECKPOINT_SECTION8 = f"{CHECKPOINT_BASE_DIR}/section8_geographic_features.parquet"

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)

# Required columns that Section 8 adds
required_columns_section8 = ["customer_merchant_distance_km", "distance_category", "city_size"]

# Verify all required columns exist
missing_cols = [col for col in required_columns_section8 if col not in train_df.columns]
if missing_cols:
    print(f"⚠️  WARNING: Missing geographical columns: {missing_cols}")
    print("  Some geographical features may not have been created.")
    print("  Checkpoint will be saved with available columns.")
else:
    print("✓ All Section 8 geographical columns present - saving section-level checkpoint...")

# Save checkpoint
try:
    train_df.write.mode("overwrite").parquet(CHECKPOINT_SECTION8)
    print(f"✓ Section 8 section-level checkpoint saved to: {CHECKPOINT_SECTION8}")
    print("✓ All geographical features are now complete!")
    print("  Note: You can now skip Section 8 in future runs by loading from this checkpoint.")
except Exception as e:
    print(f"⚠️  ERROR: Failed to save checkpoint: {e}")
    print("  Continuing without checkpoint save...")
```

**Replace With:**
```python
# ============================================================
# SAVE SECTION-LEVEL CHECKPOINT: SECTION 8 (GEOGRAPHICAL FEATURES COMPLETE)
# ============================================================

# Required columns that Section 8 adds
required_columns_section8 = ["customer_merchant_distance_km", "distance_category", "city_size"]

missing_cols = [col for col in required_columns_section8 if col not in train_df.columns]
if missing_cols:
    print(f"WARNING: Missing geographical columns: {missing_cols}")
    print("  Some geographical features may not have been created.")
    print("  Checkpoint will be saved with available columns.")
else:
    print("All Section 8 geographical columns present - saving section-level checkpoint...")

checkpoint_manager.save_checkpoint(
    train_df,
    CHECKPOINT_SECTION8,
    "Section 8 (Geographical Features Complete - Section-Level Checkpoint)"
)
print("Section 8 section-level checkpoint saved.")
```

**Changes:**
- Remove `CHECKPOINT_BASE_DIR` and `CHECKPOINT_SECTION8` redefinitions (use global from Section 1.2)
- Remove `os.makedirs(...)` (handled by `CheckpointManager` or Section 1.2)
- Remove `try/except` and raw `write.parquet` (use `checkpoint_manager.save_checkpoint`)
- Remove emojis from print statements (per user rules)

### Fix for Section 9.4

**Status:** ✅ **ALREADY FIXED** - User has already applied this fix. Section 9.4 now uses `checkpoint_manager.save_checkpoint()` correctly.

### Implementation

**File:** `notebooks/01-fraud-detection-eda.ipynb`  
**Cell:** Section 8.7 code cell (around line 5575-5605)

**Action:** Apply the Section 8.7 fix above. Section 9.4 is already correct.

---

## 2.9: Section 7 & 8 Load Checkpoint Bug (When Checkpoint Missing)

### Issue

**Location:**
- Section 7.0: Load Checkpoint code cell
- Section 8.0: Load Checkpoint code cell

**Current Implementation:**

**Section 7.0:**
```python
train_df, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path=CHECKPOINT_SECTION6,
    required_columns=required_columns_section7,
    cell_name="Section 6 (required for Section 7)"
)

if loaded_from_checkpoint:
    TOTAL_DATASET_ROWS = train_df.count()
    print(f"✓ Loaded {TOTAL_DATASET_ROWS:,} rows from checkpoint - ready for temporal analysis")
else:
    if 'train_df' not in globals() or train_df is None:
        raise ValueError(...)
    # ... rest of else block
```

**Section 8.0:**
```python
train_df, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path=CHECKPOINT_SECTION7,
    required_columns=required_columns_section8_base,
    cell_name="Section 7 (preferred for Section 8)"
)

if not loaded_from_checkpoint:
    train_df, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path=CHECKPOINT_SECTION6,
        required_columns=required_columns_section8_base,
        cell_name="Section 6 (fallback for Section 8)"
    )
# ... rest
```

**Problem:**

When `checkpoint_manager.load_checkpoint()` cannot find the checkpoint, it returns `(None, False)`. 

In Section 7.0:
- If checkpoint missing: `train_df, loaded_from_checkpoint = (None, False)`
- This **overwrites** `train_df` with `None`
- The `else` branch checks `train_df is None` and raises an error
- **The in-memory `train_df` (from running Section 6) is lost**

In Section 8.0:
- If both Section 7 and Section 6 checkpoints are missing: `train_df` gets overwritten with `None` twice
- The final `else` branch checks `'train_df' not in globals()` - but `train_df` IS in globals (it's `None`)
- Then it tries `train_df.columns` which fails because `train_df` is `None`

**Correct Pattern (from Section 9.0):**

Section 9.0 correctly uses:
```python
df_loaded, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(...)

if loaded_from_checkpoint:
    train_df = df_loaded  # Only assign when loaded
    ...
else:
    # train_df is unchanged (still in memory from previous sections)
    if 'train_df' not in globals() or train_df is None:
        raise ValueError(...)
    # ... validate and use existing train_df
```

### Fix for Section 7.0

**Current Code:**
```python
train_df, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path=CHECKPOINT_SECTION6,
    required_columns=required_columns_section7,
    cell_name="Section 6 (required for Section 7)"
)

if loaded_from_checkpoint:
    TOTAL_DATASET_ROWS = train_df.count()
    print(f"✓ Loaded {TOTAL_DATASET_ROWS:,} rows from checkpoint - ready for temporal analysis")
else:
    if 'train_df' not in globals() or train_df is None:
        raise ValueError(
            "train_df not found. Run Section 6 (Timezone Resolution) first or ensure checkpoint exists."
        )
    missing_cols = [c for c in required_columns_section7 if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}. Run Section 6 first.")
    TOTAL_DATASET_ROWS = train_df.count()
    print("✓ train_df already in memory")
```

**Replace With:**
```python
df_loaded, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path=CHECKPOINT_SECTION6,
    required_columns=required_columns_section7,
    cell_name="Section 6 (required for Section 7)"
)

if loaded_from_checkpoint:
    train_df = df_loaded
    TOTAL_DATASET_ROWS = train_df.count()
    print(f"✓ Loaded {TOTAL_DATASET_ROWS:,} rows from checkpoint - ready for temporal analysis")
else:
    if 'train_df' not in globals() or train_df is None:
        raise ValueError(
            "train_df not found. Run Section 6 (Timezone Resolution) first or ensure checkpoint exists."
        )
    missing_cols = [c for c in required_columns_section7 if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}. Run Section 6 first.")
    TOTAL_DATASET_ROWS = train_df.count()
    print("✓ train_df already in memory")
```

**Key Change:** Use `df_loaded` instead of `train_df` in the unpacking, then only assign `train_df = df_loaded` when `loaded_from_checkpoint` is `True`.

### Fix for Section 8.0

**Current Code:**
```python
# Try to load from Section 7 checkpoint (includes temporal features)
train_df, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path=CHECKPOINT_SECTION7,
    required_columns=required_columns_section8_base,
    cell_name="Section 7 (preferred for Section 8)"
)

if not loaded_from_checkpoint:
    # Fallback: Try Section 6 checkpoint
    train_df, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path=CHECKPOINT_SECTION6,
        required_columns=required_columns_section8_base,
        cell_name="Section 6 (fallback for Section 8)"
    )

if not loaded_from_checkpoint:
    # Final fallback: Check if train_df exists
    if 'train_df' not in globals():
        raise ValueError(
            "train_df not found. Please run Section 6 (Timezone Resolution) first, "
            "or ensure a checkpoint exists."
        )
    
    # Validate required columns exist
    missing_cols = [col for col in required_columns_section8_base if col not in train_df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns missing: {missing_cols}. "
            "Please run Section 6 (Timezone Resolution) first."
        )
    else:
        print("✓ train_df has required columns (not from checkpoint)")
else:
    print("✓ Loaded train_df from checkpoint - ready for geographical analysis")
```

**Replace With:**
```python
# Try to load from Section 7 checkpoint (includes temporal features)
df_loaded, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path=CHECKPOINT_SECTION7,
    required_columns=required_columns_section8_base,
    cell_name="Section 7 (preferred for Section 8)"
)

if loaded_from_checkpoint:
    train_df = df_loaded
    TOTAL_DATASET_ROWS = train_df.count()
    print(f"✓ Loaded {TOTAL_DATASET_ROWS:,} rows from checkpoint - ready for geographical analysis")
else:
    # Fallback: Try Section 6 checkpoint
    df_loaded, loaded_from_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path=CHECKPOINT_SECTION6,
        required_columns=required_columns_section8_base,
        cell_name="Section 6 (fallback for Section 8)"
    )
    
    if loaded_from_checkpoint:
        train_df = df_loaded
        TOTAL_DATASET_ROWS = train_df.count()
        print(f"✓ Loaded {TOTAL_DATASET_ROWS:,} rows from checkpoint - ready for geographical analysis")
    else:
        # Final fallback: Check if train_df exists in memory
        if 'train_df' not in globals() or train_df is None:
            raise ValueError(
                "train_df not found. Please run Section 6 (Timezone Resolution) first, "
                "or ensure a checkpoint exists."
            )
        
        # Validate required columns exist
        missing_cols = [col for col in required_columns_section8_base if col not in train_df.columns]
        if missing_cols:
            raise ValueError(
                f"Required columns missing: {missing_cols}. "
                "Please run Section 6 (Timezone Resolution) first."
            )
        TOTAL_DATASET_ROWS = train_df.count()
        print("✓ train_df already in memory - ready for geographical analysis")
```

**Key Changes:**
- Use `df_loaded` instead of `train_df` in both `load_checkpoint` calls
- Only assign `train_df = df_loaded` when `loaded_from_checkpoint` is `True`
- Add `TOTAL_DATASET_ROWS = train_df.count()` in all branches for consistency
- Fix the final `else` branch to check `train_df is None` (not just `not in globals()`)

### Implementation

**File:** `notebooks/01-fraud-detection-eda.ipynb`  
**Cells:**
- Section 7.0: Load Checkpoint code cell (around line 2751-2768)
- Section 8.0: Load Checkpoint code cell (around line 3965-3998)

**Action:** Apply the fixes above to both cells.

---

## Testing After Fixes

After applying these fixes, test the following scenarios:

1. **Fresh run (no checkpoints):** Run Sections 1-9 sequentially - should work correctly
2. **With checkpoints:** Load from checkpoints at Sections 6, 7, 8, 9 - should work correctly
3. **Mixed (some checkpoints missing):** 
   - Run Sections 1-6, then load Section 7 (no checkpoint) - should use in-memory `train_df`
   - Run Sections 1-7, then load Section 8 (no checkpoint) - should use in-memory `train_df`

---

## Priority

- **2.7 (Roadmap):** Low priority - cosmetic, doesn't affect functionality
- **2.8 (Save consistency):** Medium priority - improves consistency and maintainability
- **2.9 (Load bug):** Medium-High priority - fixes actual bug when checkpoints are missing

---

## Notes

- All fixes are **backward compatible** - they don't break existing functionality
- The fixes follow the same pattern already used in Section 9.0 (which is correct)
- Removing emojis from print statements aligns with user's "no emojis in code" rule
