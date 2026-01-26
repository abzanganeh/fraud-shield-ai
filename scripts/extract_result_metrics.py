"""
Extract key metrics from all parquet result files in data/checkpoints/results/
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json

def calculate_risk_ratio(df: pd.DataFrame, fraud_rate_col: str = 'fraud_rate_pct') -> Dict[str, float]:
    """Calculate risk ratios between highest and lowest fraud rate segments."""
    if fraud_rate_col not in df.columns:
        return {}
    
    max_fraud = df[fraud_rate_col].max()
    min_fraud = df[fraud_rate_col].min()
    
    if min_fraud > 0:
        risk_ratio = max_fraud / min_fraud
    else:
        risk_ratio = float('inf') if max_fraud > 0 else 0.0
    
    return {
        'max_fraud_rate': max_fraud,
        'min_fraud_rate': min_fraud,
        'risk_ratio': risk_ratio
    }

def extract_metrics_from_file(file_path: Path) -> Dict[str, Any]:
    """Extract key metrics from a single parquet file."""
    try:
        df = pd.read_parquet(file_path)
        
        metrics = {
            'file_name': file_path.name,
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Extract key metrics if they exist
        if 'fraud_rate_pct' in df.columns:
            metrics['fraud_rate_pct'] = {
                'min': float(df['fraud_rate_pct'].min()),
                'max': float(df['fraud_rate_pct'].max()),
                'mean': float(df['fraud_rate_pct'].mean()),
                'values': df['fraud_rate_pct'].tolist()
            }
            
            # Find segments with extreme values
            max_idx = df['fraud_rate_pct'].idxmax()
            min_idx = df['fraud_rate_pct'].idxmin()
            
            metrics['highest_fraud_segment'] = df.loc[max_idx].to_dict()
            metrics['lowest_fraud_segment'] = df.loc[min_idx].to_dict()
            
            # Calculate risk ratios
            risk_ratios = calculate_risk_ratio(df)
            metrics['risk_ratios'] = risk_ratios
            
            # Flag 100% fraud rates
            if df['fraud_rate_pct'].max() >= 100.0:
                metrics['has_100_percent_fraud'] = True
                metrics['100_percent_fraud_segments'] = df[df['fraud_rate_pct'] >= 100.0].to_dict('records')
            else:
                metrics['has_100_percent_fraud'] = False
        
        if 'total_txns' in df.columns:
            metrics['total_txns'] = {
                'min': int(df['total_txns'].min()),
                'max': int(df['total_txns'].max()),
                'sum': int(df['total_txns'].sum()),
                'mean': float(df['total_txns'].mean())
            }
        
        if 'fraud_count' in df.columns:
            metrics['fraud_count'] = {
                'min': int(df['fraud_count'].min()),
                'max': int(df['fraud_count'].max()),
                'sum': int(df['fraud_count'].sum()),
                'mean': float(df['fraud_count'].mean())
            }
        
        if 'legit_count' in df.columns:
            metrics['legit_count'] = {
                'min': int(df['legit_count'].min()),
                'max': int(df['legit_count'].max()),
                'sum': int(df['legit_count'].sum()),
                'mean': float(df['legit_count'].mean())
            }
        
        # Store full dataframe summary
        metrics['data_summary'] = df.to_dict('records')
        
        return metrics
        
    except Exception as e:
        return {
            'file_name': file_path.name,
            'error': str(e)
        }

def main():
    """Main function to extract metrics from all result files."""
    # Use absolute path from script location
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / 'data' / 'checkpoints' / 'results'
    
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist")
        return
    
    # Get all parquet files
    parquet_files = sorted(results_dir.glob('*.parquet'))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_metrics = {}
    
    for file_path in parquet_files:
        print(f"\nProcessing {file_path.name}...")
        metrics = extract_metrics_from_file(file_path)
        all_metrics[file_path.name] = metrics
        
        # Print summary
        if 'error' not in metrics:
            print(f"  Rows: {metrics.get('total_rows', 'N/A')}")
            if 'fraud_rate_pct' in metrics:
                fr = metrics['fraud_rate_pct']
                print(f"  Fraud Rate: {fr['min']:.4f}% - {fr['max']:.4f}% (mean: {fr['mean']:.4f}%)")
                if metrics.get('has_100_percent_fraud'):
                    print(f"  ⚠️  WARNING: Contains 100% fraud rate segments!")
            if 'risk_ratios' in metrics:
                rr = metrics['risk_ratios']
                print(f"  Risk Ratio: {rr.get('risk_ratio', 'N/A'):.2f}x")
        else:
            print(f"  ERROR: {metrics['error']}")
    
    # Save results to JSON
    output_file = script_dir / 'data' / 'checkpoints' / 'extracted_metrics.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Print summary of key findings
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    for file_name, metrics in all_metrics.items():
        if 'error' in metrics:
            continue
            
        if 'has_100_percent_fraud' in metrics and metrics['has_100_percent_fraud']:
            print(f"\n{file_name}:")
            print(f"  ⚠️  100% FRAUD RATE DETECTED")
            segments = metrics.get('100_percent_fraud_segments', [])
            for seg in segments[:3]:  # Show first 3
                print(f"     - {seg}")
        
        if 'risk_ratios' in metrics:
            rr = metrics['risk_ratios']
            if rr.get('risk_ratio', 0) > 10:
                print(f"\n{file_name}:")
                print(f"  ⚠️  EXTREME RISK RATIO: {rr['risk_ratio']:.2f}x")
                print(f"     Max: {rr['max_fraud_rate']:.4f}%")
                print(f"     Min: {rr['min_fraud_rate']:.4f}%")

if __name__ == '__main__':
    main()
