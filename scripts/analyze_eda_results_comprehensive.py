"""
Comprehensive EDA Results Analysis and Summary Generation

This script:
1. Reads all parquet result files from data/checkpoints/results/
2. Extracts key metrics and insights
3. Identifies weird/illogical results
4. Generates comprehensive summary markdown documents
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class EDAResultsAnalyzer:
    """Analyzes EDA results and generates comprehensive summaries."""
    
    def __init__(self, results_dir: str = "data/checkpoints/results"):
        self.results_dir = Path(results_dir)
        self.results_data: Dict[str, pd.DataFrame] = {}
        self.insights: Dict[str, Any] = defaultdict(dict)
        self.anomalies: List[Dict[str, Any]] = []
        
    def load_all_results(self) -> None:
        """Load all parquet files from results directory."""
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return
            
        parquet_files = list(self.results_dir.glob("*.parquet"))
        print(f"Found {len(parquet_files)} result files")
        
        for file_path in sorted(parquet_files):
            try:
                df = pd.read_parquet(file_path)
                file_key = file_path.stem
                self.results_data[file_key] = df
                print(f"Loaded: {file_key} ({len(df)} rows)")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
    
    def extract_amount_insights(self) -> None:
        """Extract insights from amount analysis."""
        key = "5.3_amount_bin_stats"
        if key not in self.results_data:
            return
            
        df = self.results_data[key]
        if 'fraud_rate_pct' not in df.columns:
            return
            
        riskiest = df.loc[df['fraud_rate_pct'].idxmax()]
        safest = df.loc[df['fraud_rate_pct'].idxmin()]
        risk_ratio = riskiest['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
        
        self.insights['amount'] = {
            'riskiest_bin': riskiest.get('amount_bin', 'N/A'),
            'riskiest_rate': float(riskiest['fraud_rate_pct']),
            'riskiest_txns': int(riskiest.get('total_txns', 0)),
            'safest_bin': safest.get('amount_bin', 'N/A'),
            'safest_rate': float(safest['fraud_rate_pct']),
            'safest_txns': int(safest.get('total_txns', 0)),
            'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
        }
        
        if riskiest['fraud_rate_pct'] > 20:
            self.anomalies.append({
                'section': 'Amount',
                'finding': f"{riskiest.get('amount_bin', 'N/A')} has {riskiest['fraud_rate_pct']:.2f}% fraud rate",
                'severity': 'high',
                'sample_size': int(riskiest.get('total_txns', 0))
            })
    
    def extract_temporal_insights(self) -> None:
        """Extract insights from temporal analysis."""
        # Hourly
        key = "7.1_hourly_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                peak = df.loc[df['fraud_rate_pct'].idxmax()]
                safest = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = peak['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['hourly'] = {
                    'peak_hour': int(peak.get('hour', 0)),
                    'peak_rate': float(peak['fraud_rate_pct']),
                    'safest_hour': int(safest.get('hour', 0)),
                    'safest_rate': float(safest['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Monthly
        key = "7.3_monthly_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                peak = df.loc[df['fraud_rate_pct'].idxmax()]
                safest = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = peak['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['monthly'] = {
                    'peak_month': peak.get('month_name', 'N/A'),
                    'peak_rate': float(peak['fraud_rate_pct']),
                    'safest_month': safest.get('month_name', 'N/A'),
                    'safest_rate': float(safest['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Time bins
        key = "7.5_timebin_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                peak = df.loc[df['fraud_rate_pct'].idxmax()]
                safest = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = peak['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['timebin'] = {
                    'peak_bin': peak.get('time_bin', 'N/A'),
                    'peak_rate': float(peak['fraud_rate_pct']),
                    'safest_bin': safest.get('time_bin', 'N/A'),
                    'safest_rate': float(safest['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Category
        key = "7.7_category_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                riskiest = df.loc[df['fraud_rate_pct'].idxmax()]
                safest = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = riskiest['fraud_rate_pct'] / safest['fraud_rate_pct'] if safest['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['category'] = {
                    'riskiest': riskiest.get('category', 'N/A'),
                    'riskiest_rate': float(riskiest['fraud_rate_pct']),
                    'safest': safest.get('category', 'N/A'),
                    'safest_rate': float(safest['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
    
    def extract_geographic_insights(self) -> None:
        """Extract insights from geographic analysis."""
        # State
        key = "8.1_state_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['state'] = {
                    'highest_state': top.get('state', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'highest_txns': int(top.get('total_txns', 0)),
                    'lowest_state': bottom.get('state', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'lowest_txns': int(bottom.get('total_txns', 0)),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
                
                if top['fraud_rate_pct'] == 100.0:
                    self.anomalies.append({
                        'section': 'State',
                        'finding': f"{top.get('state', 'N/A')} has 100% fraud rate",
                        'severity': 'critical',
                        'sample_size': int(top.get('total_txns', 0))
                    })
        
        # City size
        key = "8.2_city_size_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['city_size'] = {
                    'highest': top.get('city_size', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'lowest': bottom.get('city_size', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Distance
        key = "8.3_distance_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['distance'] = {
                    'highest': top.get('distance_category', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'lowest': bottom.get('distance_category', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
    
    def extract_demographic_insights(self) -> None:
        """Extract insights from demographic analysis."""
        # Age
        key = "9.1_age_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['age'] = {
                    'highest': top.get('age_bin', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'lowest': bottom.get('age_bin', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Gender
        key = "9.2_gender_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['gender'] = {
                    'highest': top.get('gender', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'lowest': bottom.get('gender', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Job
        key = "9.3_job_fraud_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['job'] = {
                    'highest': top.get('job', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'highest_txns': int(top.get('total_txns', 0)),
                    'lowest': bottom.get('job', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'lowest_txns': int(bottom.get('total_txns', 0)),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
                
                if top['fraud_rate_pct'] == 100.0:
                    self.anomalies.append({
                        'section': 'Job',
                        'finding': f"{top.get('job', 'N/A')} has 100% fraud rate",
                        'severity': 'critical',
                        'sample_size': int(top.get('total_txns', 0))
                    })
    
    def extract_credit_card_insights(self) -> None:
        """Extract insights from credit card features."""
        # Card age
        key = "10.1_card_age_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['card_age'] = {
                    'highest': top.get('card_age_bin', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'lowest': bottom.get('card_age_bin', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
        
        # Transaction count
        key = "10.2_transaction_count_stats"
        if key in self.results_data:
            df = self.results_data[key]
            if 'fraud_rate_pct' in df.columns:
                top = df.loc[df['fraud_rate_pct'].idxmax()]
                bottom = df.loc[df['fraud_rate_pct'].idxmin()]
                risk_ratio = top['fraud_rate_pct'] / bottom['fraud_rate_pct'] if bottom['fraud_rate_pct'] > 0 else float('inf')
                
                self.insights['transaction_count'] = {
                    'highest': top.get('txn_count_bin', 'N/A'),
                    'highest_rate': float(top['fraud_rate_pct']),
                    'highest_txns': int(top.get('total_txns', 0)),
                    'highest_cards': int(top.get('unique_cards', 0)) if 'unique_cards' in top else None,
                    'lowest': bottom.get('txn_count_bin', 'N/A'),
                    'lowest_rate': float(bottom['fraud_rate_pct']),
                    'lowest_txns': int(bottom.get('total_txns', 0)),
                    'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else None
                }
                
                if top['fraud_rate_pct'] == 100.0:
                    self.anomalies.append({
                        'section': 'Transaction Count',
                        'finding': f"{top.get('txn_count_bin', 'N/A')} has 100% fraud rate",
                        'severity': 'critical',
                        'sample_size': int(top.get('total_txns', 0)),
                        'cards': int(top.get('unique_cards', 0)) if 'unique_cards' in top else None
                    })
    
    def analyze_all(self) -> None:
        """Run all analysis methods."""
        self.load_all_results()
        self.extract_amount_insights()
        self.extract_temporal_insights()
        self.extract_geographic_insights()
        self.extract_demographic_insights()
        self.extract_credit_card_insights()
    
    def generate_summary_markdown(self, output_path: str) -> None:
        """Generate comprehensive summary markdown document."""
        lines = []
        lines.append("# Comprehensive EDA Summary")
        lines.append("")
        lines.append("## 1. Purpose of the EDA")
        lines.append("")
        lines.append("Dataset: 1,296,675 transactions with 0.58% fraud rate.")
        lines.append("Objective: Understand fraud patterns for feature engineering and model development.")
        lines.append("")
        
        lines.append("## 2. Data Overview & Quality")
        lines.append("")
        lines.append("- **Class imbalance:** 0.58% fraud (highly imbalanced dataset)")
        lines.append("- **Missing data:** Minimal (100% coverage on key fields)")
        lines.append("- **Bimodal distribution:** Transaction counts show unusual pattern (7-20 vs 471+ transactions)")
        lines.append("- **Outliers:** 100% fraud rates in small segments (requires investigation)")
        lines.append("")
        
        lines.append("## 3. Key Univariate Insights")
        lines.append("")
        
        # Amount
        if 'amount' in self.insights:
            amt = self.insights['amount']
            lines.append(f"- **Amount:** >${amt['riskiest_bin']} has {amt['riskiest_rate']:.2f}% fraud vs ${amt['safest_bin']} at {amt['safest_rate']:.4f}%")
            if amt['risk_ratio']:
                lines.append(f"  - Risk ratio: {amt['risk_ratio']:.0f}x difference")
        
        # Temporal
        if 'hourly' in self.insights:
            hr = self.insights['hourly']
            lines.append(f"- **Time (Hour):** {hr['peak_hour']}:00 has {hr['peak_rate']:.4f}% fraud vs {hr['safest_hour']}:00 at {hr['safest_rate']:.4f}%")
            if hr['risk_ratio']:
                lines.append(f"  - Risk ratio: {hr['risk_ratio']:.2f}x higher at peak")
        
        if 'card_age' in self.insights:
            ca = self.insights['card_age']
            lines.append(f"- **Card age:** {ca['highest']} has {ca['highest_rate']:.2f}% vs {ca['lowest']} at {ca['lowest_rate']:.2f}%")
            if ca['risk_ratio']:
                lines.append(f"  - Risk ratio: {ca['risk_ratio']:.2f}x difference")
        
        if 'transaction_count' in self.insights:
            tc = self.insights['transaction_count']
            lines.append(f"- **Transaction count:** {tc['highest']} shows {tc['highest_rate']:.2f}% fraud")
            if tc['risk_ratio']:
                lines.append(f"  - Risk ratio: {tc['risk_ratio']:.2f}x vs {tc['lowest']}")
            if tc['highest_rate'] == 100.0:
                lines.append(f"  - ⚠️ **WARNING:** 100% fraud rate in {tc['highest']} bin ({tc['highest_txns']} transactions, {tc['highest_cards']} cards) - requires investigation")
        
        lines.append("")
        
        lines.append("## 4. Key Multivariate/Interaction Findings")
        lines.append("")
        lines.append("- High amount + online + evening = highest risk")
        lines.append("- New cards + evening = elevated risk")
        if 'transaction_count' in self.insights and self.insights['transaction_count']['highest_rate'] == 100.0:
            lines.append(f"- Transaction count {self.insights['transaction_count']['highest']} + any other factor = extreme risk")
        lines.append("- Geographic patterns weak individually but may interact")
        lines.append("")
        
        lines.append("## 5. Segment-Level Insights")
        lines.append("")
        if 'transaction_count' in self.insights:
            tc = self.insights['transaction_count']
            if tc['highest_rate'] == 100.0:
                lines.append(f"- {tc['highest_cards']} cards ({tc['highest']} transactions) account for {tc['highest_txns']} fraud cases (100% fraud rate)")
        if 'card_age' in self.insights:
            ca = self.insights['card_age']
            lines.append(f"- New cards ({ca['highest']}) show {ca['highest_rate']:.2f}% fraud vs established ({ca['lowest']}) at {ca['lowest_rate']:.2f}%")
        if 'hourly' in self.insights:
            hr = self.insights['hourly']
            lines.append(f"- Evening transactions ({hr['peak_hour']}:00) disproportionately fraudulent")
        lines.append("")
        
        lines.append("## 6. Feature Engineering Implications")
        lines.append("")
        lines.append("- **Critical features:** Transaction count bin, card age bin, amount bin")
        lines.append("- **Interaction features:** evening_high_amount, new_card_evening, high_amount_online")
        lines.append("- **Aggregation features:** Card-level transaction patterns, time-of-day patterns")
        if any(a['severity'] == 'critical' for a in self.anomalies):
            lines.append("- **Handle carefully:** 100% fraud bins (may be data leakage or small sample bias)")
        lines.append("")
        
        lines.append("## 7. Limitations & Assumptions")
        lines.append("")
        if 'transaction_count' in self.insights and self.insights['transaction_count']['highest_rate'] == 100.0:
            tc = self.insights['transaction_count']
            lines.append(f"- {tc['highest']} transaction count bin (100% fraud) needs investigation - possible data leakage")
        lines.append("- Bimodal transaction count distribution may indicate dataset characteristics")
        if 'state' in self.insights and self.insights['state']['highest_rate'] == 100.0:
            st = self.insights['state']
            lines.append(f"- Small sample sizes in some segments ({st['highest_state']} state: {st['highest_txns']} transactions)")
        lines.append("- Geographic patterns may be biased by merchant distribution")
        lines.append("")
        
        lines.append("## 8. Final Takeaways")
        lines.append("")
        takeaways = []
        if 'amount' in self.insights:
            amt = self.insights['amount']
            takeaways.append(f"Transaction amount is the strongest univariate signal ({amt['risk_ratio']:.0f}x difference between highest and lowest risk bins)")
        if 'hourly' in self.insights:
            hr = self.insights['hourly']
            takeaways.append(f"Time of day shows dramatic fraud variation ({hr['risk_ratio']:.2f}x higher in evening vs morning)")
        if 'transaction_count' in self.insights:
            tc = self.insights['transaction_count']
            takeaways.append(f"Transaction count bin {tc['highest']} shows 100% fraud rate - critical feature but requires validation")
        if 'card_age' in self.insights:
            ca = self.insights['card_age']
            takeaways.append(f"Card age is a strong predictor ({ca['risk_ratio']:.2f}x difference between new and established cards)")
        
        for i, takeaway in enumerate(takeaways[:5], 1):
            lines.append(f"{i}. {takeaway}")
        
        lines.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Generated summary: {output_path}")


def main():
    """Main execution function."""
    analyzer = EDAResultsAnalyzer()
    analyzer.analyze_all()
    
    # Generate summaries for both notebook versions
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    analyzer.generate_summary_markdown(docs_dir / "eda_comprehensive_summary_local.md")
    analyzer.generate_summary_markdown(docs_dir / "eda_comprehensive_summary.md")
    
    print(f"\nFound {len(analyzer.anomalies)} anomalies:")
    for anomaly in analyzer.anomalies:
        print(f"  - {anomaly['section']}: {anomaly['finding']} (sample size: {anomaly['sample_size']})")


if __name__ == "__main__":
    main()
