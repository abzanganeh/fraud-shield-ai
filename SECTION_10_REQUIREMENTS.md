# Section 10: Credit Card Analysis - Requirements & Data Needs

**Status:** Not yet implemented  
**Mentioned in:** Section 7.6 Summary, Section 9.5 Summary  
**Dependencies:** Sections 1-9 must be completed first

---

## Overview

Section 10 will analyze fraud patterns based on **credit card characteristics:
- **Card age** (time since first transaction for that card)
- **Transaction history** (number of transactions, frequency, recency)
- **Card usage patterns** (new cards vs. established cards)

This analysis will help identify:
- Whether new cards are riskier than established cards
- Whether cards with high transaction volumes are more/less risky
- Temporal patterns specific to card age (e.g., fraud spikes for cards <30 days old)

---

## Required Data from Previous Sections

### From Section 9 Checkpoint (or in-memory `train_df`)

**Required Columns:**
- `cc_num` - Credit card number (identifier)
- `is_fraud` - Target variable
- `merchant_local_time` or `customer_local_time` - Timestamps (from Section 6)
- All other columns from previous sections (for potential interactions)

**Note:** The `cc_num` column should already be in the base dataset (Section 3). Verify it exists before starting Section 10.

---

## Analysis Components (Proposed Structure)

### 10.0 Load Checkpoint & Helper Functions
- Load from Section 9 checkpoint (or use in-memory `train_df`)
- Validate `cc_num` column exists
- Set up helper functions if needed

### 10.1 Card Age Analysis
**Hypothesis:** New cards (<30 days old) may have higher fraud rates

**What to calculate:**
- First transaction date per `cc_num` (minimum timestamp per card)
- Card age = current transaction date - first transaction date
- Card age bins: <7 days, 7-30 days, 30-90 days, 90-180 days, 180+ days
- Fraud rate by card age bin

**Expected outputs:**
- Card age distribution
- Fraud rate by age bin
- Risk ratio (newest vs. oldest cards)

### 10.2 Transaction History Analysis
**Hypothesis:** Cards with very few or very many transactions may show different fraud patterns

**What to calculate:**
- Transaction count per `cc_num` (group by card, count transactions)
- Transaction frequency (transactions per day/week for active cards)
- Recency (days since last transaction)
- Fraud rate by transaction count bins (e.g., 1-5, 6-20, 21-50, 51-100, 100+)

**Expected outputs:**
- Transaction count distribution
- Fraud rate by count bin
- Risk patterns (low-volume vs. high-volume cards)

### 10.3 Card Usage Patterns
**Hypothesis:** Cards with unusual usage patterns (sudden spike, long inactivity) may be riskier

**What to calculate:**
- Cards with sudden transaction spikes (e.g., >5x normal rate)
- Cards with long inactivity periods (e.g., >90 days, then sudden activity)
- Fraud rate for these patterns

**Expected outputs:**
- Pattern identification counts
- Fraud rates for each pattern
- Risk assessment

### 10.4 Key Findings Summary
- Summarize card age patterns
- Summarize transaction history patterns
- Feature engineering recommendations
- Priority assessment (high/medium/low)

### 10.5 Save Section-Level Checkpoint
- Save `train_df` with any new card-related features (card_age, transaction_count, etc.)
- Use `checkpoint_manager.save_checkpoint()` with global `CHECKPOINT_SECTION10`

---

## Data Validation Needed

Before starting Section 10, verify:

1. **`cc_num` column exists:**
   ```python
   if "cc_num" not in train_df.columns:
       raise ValueError("cc_num column not found. Check dataset schema.")
   ```

2. **`cc_num` coverage:**
   - Check for NULL/missing values
   - Count unique cards
   - Validate data quality (no duplicates, valid format)

3. **Timestamp availability:**
   - Ensure `merchant_local_time` or `customer_local_time` exists (from Section 6)
   - Verify timestamps are valid for calculating card age

---

## Expected Results from Kaggle Run

To complete Section 10, you'll need to run the notebook and capture:

### From 10.1 (Card Age Analysis):
- **Card age distribution:** Count of cards in each age bin
- **Fraud rate by age bin:** e.g., "<7 days: 2.5%", "180+ days: 0.4%"
- **Risk ratio:** e.g., "Newest cards (<7 days) are 6.25x riskier than oldest cards (180+ days)"

### From 10.2 (Transaction History):
- **Transaction count distribution:** e.g., "Cards with 1-5 transactions: 450,000 cards"
- **Fraud rate by count bin:** e.g., "1-5 transactions: 0.8%", "100+ transactions: 0.3%"
- **Risk patterns:** e.g., "Low-volume cards (1-5 txns) are 2.67x riskier than high-volume (100+)"

### From 10.3 (Usage Patterns):
- **Pattern counts:** e.g., "Cards with sudden spikes: 12,345", "Cards with long inactivity: 8,901"
- **Fraud rates:** e.g., "Sudden spike pattern: 3.2% fraud rate", "Long inactivity: 1.8% fraud rate"

### Summary Statistics:
- Overall card count
- Average transactions per card
- Average card age
- Key risk ratios

---

## Implementation Checklist

- [ ] Verify `cc_num` column exists in dataset
- [ ] Create Section 10 structure (10.0-10.5)
- [ ] Implement card age calculation (first transaction date per card)
- [ ] Implement transaction count per card
- [ ] Create age bins and count bins
- [ ] Aggregate fraud rates by bins
- [ ] Create visualizations (if needed)
- [ ] Write Key Findings Summary with quantitative results
- [ ] Add Section 10 checkpoint save
- [ ] Update Section 9.5 "Next Section" reference (already says "10.0 - Credit Card Analysis")
- [ ] Run on Kaggle and capture outputs
- [ ] Update Key Findings with actual results

---

## Code Pattern to Follow

Follow the same pattern as Sections 7, 8, 9:

1. **Load checkpoint** (Section 9) with validation
2. **Data validation** using `validate_column_coverage()` for `cc_num`
3. **Feature engineering** (card age, transaction count) using Spark operations
4. **Aggregation** using `aggregate_fraud_by_dimension()` helper
5. **Results persistence** using `results_manager.save_dataframe()` for saved results
6. **Key Findings** markdown with quantitative results
7. **Checkpoint save** using `checkpoint_manager.save_checkpoint()`

---

## Notes

- **Card age** requires calculating the **minimum timestamp per `cc_num`** (first transaction)
- **Transaction count** requires **grouping by `cc_num` and counting**
- Both operations are Spark aggregations - use `groupBy().agg()` or `groupBy().count()`
- Consider **minimum transaction thresholds** for statistical significance
- **New cards** (<7 or <30 days) are likely to show higher fraud rates (common pattern)
- **Low-volume cards** (1-5 transactions) may also be riskier (fraudsters testing cards)

---

## Dependencies

**Must complete first:**
- Section 6: Timezone resolution (for accurate timestamps)
- Section 9: Demographics analysis (to have latest `train_df` with all features)

**Optional but helpful:**
- Section 7: Temporal patterns (for interaction analysis later)
- Section 8: Geographical patterns (for interaction analysis later)

---

## Future Enhancements (Post-Section 10)

After Section 10, consider:
- **Interaction features:** Card age × time of day, Card age × amount, etc.
- **Sequential patterns:** Transaction sequences per card (requires more complex analysis)
- **Card velocity:** Transactions per hour/day for each card
- **Cross-card patterns:** Multiple cards used by same customer (if customer ID available)
