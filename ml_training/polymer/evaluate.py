"""
Evaluate and compare original vs tuned XGBoost models on the test set.
"""

import json
import argparse
from pathlib import Path
import h5py
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report

SPECTRAL_BANDS = [
    "nm440", "nm490", "nm560", "nm665", "nm705",
    "nm740", "nm783", "nm842", "nm865",
    "nm1600", "nm2200",
]

def load_test_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        tbl = f["test"]["table"][:]
        dtype_names = set(tbl.dtype.names)
        bands = [b for b in SPECTRAL_BANDS if b in dtype_names]
        X = np.stack([tbl[b].astype(np.float32) for b in bands], axis=1)
        y_raw = np.array([c.decode() if isinstance(c, bytes) else c for c in tbl["Class"]])
    return X, y_raw, bands

def compute_spectral_indices(X, bands):
    b_idx = {b: i for i, b in enumerate(bands)}
    eps = 1e-8
    
    b04 = X[:, b_idx["nm665"]] if "nm665" in b_idx else np.zeros(X.shape[0])
    b06 = X[:, b_idx["nm740"]] if "nm740" in b_idx else np.zeros(X.shape[0])
    b08 = X[:, b_idx["nm842"]] if "nm842" in b_idx else np.zeros(X.shape[0])
    b8a = X[:, b_idx["nm865"]] if "nm865" in b_idx else np.zeros(X.shape[0])
    b11 = X[:, b_idx["nm1600"]] if "nm1600" in b_idx else np.zeros(X.shape[0])

    pi = (b08 - b04) / (b08 + b04 + eps)
    sr = b11 / (b08 + eps)
    nsi = (b8a - b11) / (b8a + b11 + eps)
    
    interpolation = (832 - 665) / (1610 - 665 + eps)
    fdi = b08 - (b06 + (b11 - b06) * interpolation)

    new_features = np.stack([pi, sr, nsi, fdi], axis=1)
    new_bands = ["PI", "SR", "NSI", "FDI"]
    X_enhanced = np.hstack([X, new_features])
    return X_enhanced, bands + new_bands

def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost models on the test set.")
    parser.add_argument("--h5_path", default=str(Path(__file__).parent / "dataset.h5"), help="Path to dataset.h5")
    parser.add_argument("--models_dir", default=r"d:\DSU-DEVHACK-3.0\models\production", help="Directory containing the models")
    parser.add_argument("--output_dir", default=r"d:\DSU-DEVHACK-3.0\models\production", help="Directory to save the evaluation JSON")
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_model_path = models_dir / "polymer_xgb_model_orig.json"
    orig_feat_path  = models_dir / "polymer_feature_names_orig.json"
    tuned_model_path = models_dir / "polymer_xgb_model.pkl"
    tuned_feat_path  = models_dir / "polymer_feature_names.json"
    label_map_path   = models_dir / "polymer_label_map.json"

    print(f"Loading test data from {h5_path} ...")
    X_test, y_test_raw, orig_bands = load_test_split(h5_path)
    X_test_tuned, tuned_bands = compute_spectral_indices(X_test, orig_bands)
    
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    
    y_test = np.array([label_map[c] for c in y_test_raw])
    # Ensure classes are printed in index order
    classes = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    debris_idx = label_map["Marine Debris"]
    
    print(f"Test Set Size: {len(y_test):,} pixels")

    # ── Evaluate Original Model ───────────────────────────────────────────
    has_orig = orig_model_path.exists()
    if has_orig:
        print("\n" + "="*70)
        print("  MODEL 1: ORIGINAL (Raw Bands Only)")
        print("="*70)
        orig_model = xgb.Booster()
        orig_model.load_model(orig_model_path)
        with open(orig_feat_path, "r") as f:
            orig_feat = json.load(f)
            
        dtest_orig = xgb.DMatrix(X_test, feature_names=orig_feat)
        preds_orig = orig_model.predict(dtest_orig)
        
        print(classification_report(y_test, preds_orig, target_names=classes, digits=3, zero_division=0))
    
    # ── Evaluate Tuned Model ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("  MODEL 2: TUNED (Optuna + Spectral Indices)")
    print("="*70)
    tuned_model = joblib.load(tuned_model_path)
    with open(tuned_feat_path, "r") as f:
        tuned_feat = json.load(f)
        
    dtest_tuned = xgb.DMatrix(X_test_tuned, feature_names=tuned_feat)
    preds_tuned = tuned_model.predict(dtest_tuned)
    
    report_str = classification_report(y_test, preds_tuned, target_names=classes, digits=3, zero_division=0)
    print(report_str)
    
    report_dict = classification_report(y_test, preds_tuned, target_names=classes, digits=3, zero_division=0, output_dict=True)
    eval_out_path = output_dir / "polymer_xgb_model_eval.json"
    with open(eval_out_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"\nEvaluation scores saved to -> {eval_out_path}")

    # ── Head to Head Comparison ───────────────────────────────────────────
    if has_orig:
        def get_metrics(preds):
            debris_mask = (y_test == debris_idx)
            non_debris_mask = ~debris_mask
            
            recall = (preds[debris_mask] == debris_idx).mean()
            precision = (y_test[preds == debris_idx] == debris_idx).mean()
            f1 = 2 * (precision * recall) / (precision + recall)
            fpr = (preds[non_debris_mask] == debris_idx).mean()
            return precision, recall, f1, fpr

        orig_p, orig_r, orig_f1, orig_fpr = get_metrics(preds_orig)
        tuned_p, tuned_r, tuned_f1, tuned_fpr = get_metrics(preds_tuned)

        print("\n" + "="*70)
        print("  HEAD TO HEAD: MARINE DEBRIS CLASS")
        print("="*70)
        print(f"               | {'ORIGINAL':<15} | {'TUNED':<15} | {'IMPROVEMENT':<15}")
        print("-" * 70)
        print(f"  Precision    | {orig_p:.3f}           | {tuned_p:.3f}           | {tuned_p - orig_p:+.3f}")
        print(f"  Recall       | {orig_r:.3f}           | {tuned_r:.3f}           | {tuned_r - orig_r:+.3f}")
        print(f"  F1-Score     | {orig_f1:.3f}           | {tuned_f1:.3f}           | {tuned_f1 - orig_f1:+.3f}")
        print(f"  FPR (Noise)  | {orig_fpr:.4f}          | {tuned_fpr:.4f}          | {tuned_fpr - orig_fpr:+.4f}")
        print("="*70)


if __name__ == "__main__":
    main()
