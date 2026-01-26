"""
Script to analyze EDA result files and extract key metrics.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json


def read_result_files(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Read all parquet result files and return as dictionary."""
    results = {}
    for parquet_file in sorted(results_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_file)
            results[parquet_file.name] = df
        except Exception as e:
            print(f"Error reading {parquet_file.name}: {e}")
    return results


def extract_metrics(df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
    """Extract key metrics from a result dataframe."""
    metrics = {
        "file": file_name,
        "columns": list(df.columns),
        "row_count": len(df),
    }
    
    # Try to extract fraud-related metrics
    if "fraud_rate_pct" in df.columns:
        metrics["fraud_rate_pct"] = df["fraud_rate_pct"].tolist()
        metrics["min_fraud_rate"] = float(df["fraud_rate_pct"].min())
        metrics["max_fraud_rate"] = float(df["fraud_rate_pct"].max())
        metrics["mean_fraud_rate"] = float(df["fraud_rate_pct"].mean())
    
    if "total_txns" in df.columns:
        metrics["total_txns"] = df["total_txns"].tolist()
        metrics["total_transactions"] = int(df["total_txns"].sum())
    
    if "fraud_count" in df.columns:
        metrics["fraud_count"] = df["fraud_count"].tolist()
        metrics["total_fraud"] = int(df["fraud_count"].sum())
    
    if "risk_ratio" in df.columns:
        metrics["risk_ratio"] = df["risk_ratio"].tolist()
        metrics["max_risk_ratio"] = float(df["risk_ratio"].max())
        metrics["min_risk_ratio"] = float(df["risk_ratio"].min())
    
    # Store full dataframe summary (convert to JSON-serializable format)
    data_records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                record[col] = None
            elif isinstance(val, (pd.Timestamp, pd.DatetimeTZDtype)):
                record[col] = str(val)
            else:
                try:
                    record[col] = float(val) if isinstance(val, (int, float)) else str(val)
                except:
                    record[col] = str(val)
        data_records.append(record)
    metrics["data"] = data_records
    
    return metrics


def identify_weird_findings(all_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify suspicious findings: 100% fraud rates, extreme ratios, etc."""
    weird_findings = []
    
    for file_name, metrics in all_metrics.items():
        # Check for 100% fraud rates
        if "fraud_rate_pct" in metrics:
            fraud_rates = metrics["fraud_rate_pct"]
            for i, rate in enumerate(fraud_rates):
                if rate >= 100.0:
                    finding = {
                        "file": file_name,
                        "type": "100%_fraud_rate",
                        "fraud_rate": rate,
                        "index": i,
                    }
                    if "data" in metrics and i < len(metrics["data"]):
                        finding["details"] = metrics["data"][i]
                    weird_findings.append(finding)
        
        # Check for extreme risk ratios
        if "risk_ratio" in metrics:
            risk_ratios = metrics["risk_ratio"]
            for i, ratio in enumerate(risk_ratios):
                if ratio > 10.0:
                    finding = {
                        "file": file_name,
                        "type": "extreme_risk_ratio",
                        "risk_ratio": ratio,
                        "index": i,
                    }
                    if "data" in metrics and i < len(metrics["data"]):
                        finding["details"] = metrics["data"][i]
                    weird_findings.append(finding)
        
        # Check for very small sample sizes with high fraud rates
        if "total_txns" in metrics and "fraud_rate_pct" in metrics:
            txns = metrics["total_txns"]
            rates = metrics["fraud_rate_pct"]
            for i, (txn_count, rate) in enumerate(zip(txns, rates)):
                if txn_count < 100 and rate > 5.0:
                    finding = {
                        "file": file_name,
                        "type": "small_sample_high_fraud",
                        "transaction_count": txn_count,
                        "fraud_rate": rate,
                        "index": i,
                    }
                    if "data" in metrics and i < len(metrics["data"]):
                        finding["details"] = metrics["data"][i]
                    weird_findings.append(finding)
    
    return weird_findings


def main():
    """Main execution function."""
    results_dir = Path(__file__).parent.parent / "data" / "checkpoints" / "results"
    
    print(f"Reading result files from {results_dir}")
    all_results = read_result_files(results_dir)
    print(f"Found {len(all_results)} result files")
    
    all_metrics = {}
    for file_name, df in all_results.items():
        print(f"\nProcessing {file_name}...")
        metrics = extract_metrics(df, file_name)
        all_metrics[file_name] = metrics
        print(f"  Rows: {metrics['row_count']}, Columns: {metrics['columns']}")
    
    weird_findings = identify_weird_findings(all_metrics)
    
    output = {
        "all_metrics": all_metrics,
        "weird_findings": weird_findings,
    }
    
    output_file = Path(__file__).parent.parent / "data" / "checkpoints" / "eda_analysis_results.json"
    
    def json_serializer(obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            return str(obj)
        if pd.isna(obj):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=json_serializer)
    
    print(f"\n\nAnalysis complete!")
    print(f"Found {len(weird_findings)} suspicious findings")
    print(f"Results saved to {output_file}")
    
    return all_metrics, weird_findings


if __name__ == "__main__":
    main()
