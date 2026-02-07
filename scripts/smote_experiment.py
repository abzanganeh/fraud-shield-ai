"""
Comprehensive SMOTE Experiment
Tests different SMOTE variants and sampling strategies to find optimal configuration.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any

# SMOTE variants
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

# XGBoost for quick evaluation
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# Paths
PROJECT_ROOT = Path("/home/alireza/Desktop/projects/fraud-shield-ai")
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed train/val/test data."""
    print("Loading preprocessed data...")
    
    train_df = pd.read_parquet(DATA_DIR / "train_preprocessed.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_preprocessed.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_preprocessed.parquet")
    
    # Extract features and labels
    feature_cols = [c for c in train_df.columns if c != 'is_fraud']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['is_fraud'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['is_fraud'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['is_fraud'].values
    
    print(f"  Train: {X_train.shape[0]:,} samples ({y_train.mean():.4%} fraud)")
    print(f"  Val: {X_val.shape[0]:,} samples ({y_val.mean():.4%} fraud)")
    print(f"  Test: {X_test.shape[0]:,} samples ({y_test.mean():.4%} fraud)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str
) -> Dict[str, float]:
    """Train XGBoost and evaluate on val/test sets."""
    
    # Calculate class weight from TRAINING data (after resampling)
    fraud_rate = y_train.mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate if fraud_rate > 0 else 1.0
    
    print(f"\n  Training XGBoost (fraud_rate={fraud_rate:.4%}, scale_pos_weight={scale_pos_weight:.2f})...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        device='cuda',
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Find optimal threshold on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_val_f1 = 0
    
    for thresh in np.linspace(0.1, 0.99, 50):
        val_preds = (val_probs >= thresh).astype(int)
        val_f1 = f1_score(y_val, val_preds)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = thresh
    
    # Evaluate on test set with optimal threshold
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    
    results = {
        'experiment': experiment_name,
        'train_samples': len(y_train),
        'train_fraud_rate': y_train.mean(),
        'threshold': best_threshold,
        'val_f1': best_val_f1,
        'test_f1': f1_score(y_test, test_preds),
        'test_precision': precision_score(y_test, test_preds),
        'test_recall': recall_score(y_test, test_preds),
        'test_roc_auc': roc_auc_score(y_test, test_probs),
        'test_pr_auc': average_precision_score(y_test, test_probs)
    }
    
    print(f"  Val F1: {results['val_f1']:.4f}, Test F1: {results['test_f1']:.4f}, Test ROC-AUC: {results['test_roc_auc']:.4f}")
    
    return results


def run_experiments():
    """Run all SMOTE experiments."""
    
    print("=" * 70)
    print("COMPREHENSIVE SMOTE EXPERIMENT")
    print("=" * 70)
    
    # Load original data
    X_train_orig, y_train_orig, X_val, y_val, X_test, y_test = load_data()
    
    results_list = []
    
    # Experiment 0: Baseline (No SMOTE)
    print("\n" + "=" * 70)
    print("EXPERIMENT 0: No SMOTE (Baseline)")
    print("=" * 70)
    results = evaluate_model(X_train_orig, y_train_orig, X_val, y_val, X_test, y_test, "No SMOTE (Baseline)")
    results_list.append(results)
    
    # SMOTE configurations to test
    smote_configs = [
        # Conservative sampling strategies
        ("SMOTE_0.01", SMOTE(sampling_strategy=0.01, k_neighbors=5, random_state=42)),  # 1% fraud
        ("SMOTE_0.02", SMOTE(sampling_strategy=0.02, k_neighbors=5, random_state=42)),  # 2% fraud
        ("SMOTE_0.05", SMOTE(sampling_strategy=0.05, k_neighbors=5, random_state=42)),  # 5% fraud
        ("SMOTE_0.10", SMOTE(sampling_strategy=0.10, k_neighbors=5, random_state=42)),  # 10% fraud (original)
        
        # Different k_neighbors
        ("SMOTE_0.05_k3", SMOTE(sampling_strategy=0.05, k_neighbors=3, random_state=42)),
        ("SMOTE_0.05_k7", SMOTE(sampling_strategy=0.05, k_neighbors=7, random_state=42)),
        
        # SMOTE variants
        ("BorderlineSMOTE_0.05", BorderlineSMOTE(sampling_strategy=0.05, k_neighbors=5, random_state=42)),
        ("ADASYN_0.05", ADASYN(sampling_strategy=0.05, n_neighbors=5, random_state=42)),
        
        # Hybrid methods (SMOTE + cleaning)
        ("SMOTEENN_0.05", SMOTEENN(sampling_strategy=0.05, random_state=42)),
        ("SMOTETomek_0.05", SMOTETomek(sampling_strategy=0.05, random_state=42)),
    ]
    
    for exp_num, (name, sampler) in enumerate(smote_configs, start=1):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT {exp_num}: {name}")
        print("=" * 70)
        
        try:
            print(f"  Applying {name}...")
            X_resampled, y_resampled = sampler.fit_resample(X_train_orig, y_train_orig)
            print(f"  Before: {len(y_train_orig):,} samples ({y_train_orig.mean():.4%} fraud)")
            print(f"  After: {len(y_resampled):,} samples ({y_resampled.mean():.4%} fraud)")
            
            results = evaluate_model(X_resampled, y_resampled, X_val, y_val, X_test, y_test, name)
            results_list.append(results)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by test F1
    results_df = results_df.sort_values('test_f1', ascending=False)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (sorted by Test F1)")
    print("=" * 70)
    
    display_cols = ['experiment', 'train_fraud_rate', 'val_f1', 'test_f1', 'test_precision', 'test_recall', 'test_roc_auc']
    print(results_df[display_cols].to_string(index=False))
    
    # Save results
    output_path = PROJECT_ROOT / "results" / "smote_experiment_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Find best configuration
    best_row = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Experiment: {best_row['experiment']}")
    print(f"  Train Fraud Rate: {best_row['train_fraud_rate']:.4%}")
    print(f"  Val F1: {best_row['val_f1']:.4f}")
    print(f"  Test F1: {best_row['test_f1']:.4f}")
    print(f"  Test Precision: {best_row['test_precision']:.4f}")
    print(f"  Test Recall: {best_row['test_recall']:.4f}")
    print(f"  Test ROC-AUC: {best_row['test_roc_auc']:.4f}")
    
    return results_df


if __name__ == "__main__":
    run_experiments()
