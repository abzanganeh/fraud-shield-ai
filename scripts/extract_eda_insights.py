#!/usr/bin/env python3
"""
Extract KEY INSIGHTS from EDA notebooks and parquet result files.
"""
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas not available, will extract from notebooks only")
    pd = None

def extract_parquet_insights():
    """Extract insights from parquet result files."""
    if pd is None:
        return {}
    
    results_dir = Path("data/checkpoints/results")
    insights = {}
    
    files_to_read = {
        "5.3_amount_bin_stats.parquet": "amount",
        "7.1_hourly_fraud_stats.parquet": "hourly",
        "7.2_daily_fraud_stats.parquet": "daily",
        "7.3_monthly_fraud_stats.parquet": "monthly",
        "7.4_weekend_fraud_stats.parquet": "weekend",
        "7.5_timebin_fraud_stats.parquet": "timebin",
        "7.7_category_fraud_stats.parquet": "category",
        "8.1_state_fraud_stats.parquet": "state",
        "8.2_city_size_fraud_stats.parquet": "city_size",
        "8.3_distance_fraud_stats.parquet": "distance",
        "8.4_city_fraud_stats.parquet": "city",
        "8.4_merchant_location_stats.parquet": "merchant_location",
        "8.5_merchant_fraud_stats.parquet": "merchant",
        "9.1_age_fraud_stats.parquet": "age",
        "9.2_gender_fraud_stats.parquet": "gender",
        "9.3_job_fraud_stats.parquet": "job",
        "10.1_card_age_stats.parquet": "card_age",
        "10.2_transaction_count_stats.parquet": "transaction_count",
    }
    
    for filename, key in files_to_read.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                insights[key] = df.to_dict('records')
            except Exception as e:
                print(f"Error reading {filename}: {e}", file=sys.stderr)
    
    return insights

def analyze_insights(insights):
    """Analyze insights and identify weird findings."""
    findings = {
        "overall_fraud_rate": None,
        "weird_findings": [],
        "extreme_ratios": [],
        "empty_bins": [],
        "section_insights": {}
    }
    
    if pd is None:
        return findings
    
    # Section 4: Overall fraud rate
    if "amount" in insights:
        amount_df = pd.DataFrame(insights["amount"])
        total_fraud = amount_df["fraud_count"].sum()
        total_txns = amount_df["total_txns"].sum()
        findings["overall_fraud_rate"] = (total_fraud / total_txns) * 100 if total_txns > 0 else 0
    
    # Section 5: Amount analysis
    if "amount" in insights:
        amount_df = pd.DataFrame(insights["amount"])
        riskiest = amount_df.loc[amount_df['fraud_rate_pct'].idxmax()]
        safest = amount_df.loc[amount_df['fraud_rate_pct'].idxmin()]
        risk_ratio = riskiest['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
        
        findings["section_insights"]["amount"] = {
            "riskiest": riskiest.to_dict(),
            "safest": safest.to_dict(),
            "risk_ratio": risk_ratio
        }
        
        if riskiest['fraud_rate_pct'] == 100.0:
            findings["weird_findings"].append({
                "section": "5.3",
                "finding": f"100% fraud rate in amount bin: {riskiest.get('amount_bin', 'unknown')}",
                "details": riskiest.to_dict()
            })
    
    # Section 7: Temporal patterns
    if "hourly" in insights:
        hourly_df = pd.DataFrame(insights["hourly"])
        peak = hourly_df.loc[hourly_df['fraud_rate_pct'].idxmax()]
        safest = hourly_df.loc[hourly_df['fraud_rate_pct'].idxmin()]
        risk_ratio = peak['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
        
        findings["section_insights"]["hourly"] = {
            "peak": peak.to_dict(),
            "safest": safest.to_dict(),
            "risk_ratio": risk_ratio
        }
        
        if risk_ratio > 30:
            findings["extreme_ratios"].append({
                "section": "7.1",
                "metric": "hourly",
                "ratio": risk_ratio,
                "details": f"Peak: {peak.get('hour')} ({peak['fraud_rate_pct']:.4f}%), Safest: {safest.get('hour')} ({safest['fraud_rate_pct']:.4f}%)"
            })
    
    # Section 8: Geographic patterns
    if "state" in insights:
        state_df = pd.DataFrame(insights["state"])
        top = state_df.loc[state_df['fraud_rate_pct'].idxmax()]
        bottom = state_df.loc[state_df['fraud_rate_pct'].idxmin()]
        risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
        
        findings["section_insights"]["state"] = {
            "highest": top.to_dict(),
            "lowest": bottom.to_dict(),
            "risk_ratio": risk_ratio
        }
        
        if top['fraud_rate_pct'] == 100.0:
            findings["weird_findings"].append({
                "section": "8.1",
                "finding": f"100% fraud rate in state: {top.get('state', 'unknown')}",
                "details": top.to_dict()
            })
    
    # Section 9: Demographics
    if "job" in insights:
        job_df = pd.DataFrame(insights["job"])
        for _, row in job_df.iterrows():
            if row['fraud_rate_pct'] == 100.0 and row['total_txns'] < 100:
                findings["weird_findings"].append({
                    "section": "9.3",
                    "finding": f"100% fraud rate in job: {row.get('job', 'unknown')} (low volume: {row['total_txns']} txns)",
                    "details": row.to_dict()
                })
    
    # Section 10: Credit card features
    if "transaction_count" in insights:
        txn_df = pd.DataFrame(insights["transaction_count"])
        for _, row in txn_df.iterrows():
            if row['fraud_rate_pct'] == 100.0:
                findings["weird_findings"].append({
                    "section": "10.2",
                    "finding": f"100% fraud rate in transaction count bin: {row.get('transaction_count_bin', 'unknown')}",
                    "details": row.to_dict()
                })
        
        if len(txn_df) > 0:
            riskiest = txn_df.loc[txn_df['fraud_rate_pct'].idxmax()]
            safest = txn_df.loc[txn_df['fraud_rate_pct'].idxmin()]
            risk_ratio = riskiest['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
            
            findings["section_insights"]["transaction_count"] = {
                "riskiest": riskiest.to_dict(),
                "safest": safest.to_dict(),
                "risk_ratio": risk_ratio
            }
            
            if risk_ratio > 100:
                findings["extreme_ratios"].append({
                    "section": "10.2",
                    "metric": "transaction_count",
                    "ratio": risk_ratio,
                    "details": riskiest.to_dict()
                })
    
    if "card_age" in insights:
        age_df = pd.DataFrame(insights["card_age"])
        if len(age_df) > 0:
            riskiest = age_df.loc[age_df['fraud_rate_pct'].idxmax()]
            safest = age_df.loc[age_df['fraud_rate_pct'].idxmin()]
            risk_ratio = riskiest['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
            
            findings["section_insights"]["card_age"] = {
                "riskiest": riskiest.to_dict(),
                "safest": safest.to_dict(),
                "risk_ratio": risk_ratio
            }
            
            if risk_ratio > 10:
                findings["extreme_ratios"].append({
                    "section": "10.1",
                    "metric": "card_age",
                    "ratio": risk_ratio,
                    "details": f"Riskiest: {riskiest.get('card_age_bin', 'unknown')} ({riskiest['fraud_rate_pct']:.4f}%), Safest: {safest.get('card_age_bin', 'unknown')} ({safest['fraud_rate_pct']:.4f}%)"
                })
    
    return findings

if __name__ == "__main__":
    insights = extract_parquet_insights()
    findings = analyze_insights(insights)
    
    output = {
        "raw_data": insights,
        "analysis": findings
    }
    
    print(json.dumps(output, indent=2, default=str))
