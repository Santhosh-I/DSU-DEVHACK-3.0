"""
Stage 4 — Polymer Classification Model (XGBoost Tuned with Optuna)
========================================
Trains an XGBoost model on MARIDA pixel-level spectral signatures
(11 Sentinel-2 bands + 4 indices) to classify Marine Debris
vs. all other sea-surface classes using GPU acceleration and Optuna.

Output:
    polymer_xgb_model_tuned.json   — tuned classifier (XGBoost)
    polymer_label_map_tuned.json   — class name → integer mapping
    polymer_feature_names_tuned.json — ordered list of band names used
"""

import json
from pathlib import Path

import h5py
import numpy as np
import xgboost as xgb
import optuna
import joblib
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
H5_PATH  = Path(__file__).parent / "dataset.h5"
OUT_DIR  = Path(r"D:\DSU-DEVHACK-3.0\models\polymer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = OUT_DIR / "polymer_xgb_model_tuned.pkl"
LABEL_MAP_OUT = OUT_DIR / "polymer_label_map_tuned.json"
FEAT_OUT  = OUT_DIR / "polymer_feature_names_tuned.json"

# The 11 Sentinel-2 spectral band columns present in the h5
SPECTRAL_BANDS = [
    "nm440", "nm490", "nm560", "nm665", "nm705",
    "nm740", "nm783", "nm842", "nm865",
    "nm1600", "nm2200",
]

PLASTIC_CLASSES = {"Marine Debris"}

# ── Feature Engineering ──────────────────────────────────────────────────────
def compute_spectral_indices(X, bands):
    """Compute and append PI, SR, NSI, FDI to the feature matrix X."""
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
    
    # FDI = B08 - (B06 + (B11 - B06) * ((832 - 665) / (1610 - 665)))
    interpolation = (832 - 665) / (1610 - 665 + eps)
    fdi = b08 - (b06 + (b11 - b06) * interpolation)

    new_features = np.stack([pi, sr, nsi, fdi], axis=1)
    new_bands = ["PI", "SR", "NSI", "FDI"]
    
    X_enhanced = np.hstack([X, new_features])
    return X_enhanced, bands + new_bands


def load_split(f: h5py.File, split: str):
    """Load spectral features + class labels from one h5 split."""
    tbl = f[split]["table"][:]
    dtype_names = set(tbl.dtype.names)

    bands = [b for b in SPECTRAL_BANDS if b in dtype_names]
    X = np.stack([tbl[b].astype(np.float32) for b in bands], axis=1)
    y_raw = np.array([c.decode() if isinstance(c, bytes) else c for c in tbl["Class"]])

    X, enhanced_bands = compute_spectral_indices(X, bands)
    return X, y_raw, enhanced_bands


def main():
    print(f"Loading MARIDA spectral signatures from {H5_PATH} ...")
    with h5py.File(H5_PATH, "r") as f:
        X_train, y_train_raw, bands = load_split(f, "train")
        X_val,   y_val_raw,   _     = load_split(f, "val")
        X_test,  y_test_raw,  _     = load_split(f, "test")

    print(f"  Train: {X_train.shape[0]:,} pixels  |  {len(bands)} bands")
    print(f"  Bands: {bands}")

    # Label encoding
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_val_raw, y_test_raw]))
    
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    
    classes = list(le.classes_)
    num_classes = len(classes)
    debris_idx = le.transform(["Marine Debris"])[0]

    dval = xgb.DMatrix(X_val, label=y_val, feature_names=bands)

    # ── Optuna Tuning ────────────────────────────────────────────────────────
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 4, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3, log=True)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        debris_weight = trial.suggest_float("debris_weight", 1.0, 15.0)

        # Update weights dynamically
        sample_weights = np.ones(len(y_train), dtype=np.float32)
        sample_weights[y_train == debris_idx] = debris_weight
        dtrain_w = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, feature_names=bands)

        params = {
            "objective": "multi:softmax",
            "num_class": num_classes,
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "colsample_bytree": colsample_bytree,
            "eval_metric": "mlogloss",
            "seed": 42
        }

        # Train a small number of rounds for fast tuning
        model = xgb.train(
            params,
            dtrain_w,
            num_boost_round=100,
            evals=[(dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        
        y_val_bin = (y_val == debris_idx).astype(int)
        preds_bin = (preds == debris_idx).astype(int)
        
        f1 = f1_score(y_val_bin, preds_bin, zero_division=0)
        return f1

    print("\nStarting Optuna Hyperparameter Tuning...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("\nBest trial:")
    print(f"  Value (Marine Debris F1): {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # ── Train Final Model with Best Params ──────────────────────────────────
    print("\nTraining Final Model with Best Parameters on Train+Val...")
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    
    best_weight = study.best_params.pop("debris_weight")
    sample_weights_all = np.ones(len(y_all), dtype=np.float32)
    sample_weights_all[y_all == debris_idx] = best_weight
    
    dall_w = xgb.DMatrix(X_all, label=y_all, weight=sample_weights_all, feature_names=bands)
    
    final_params = {
        "objective": "multi:softmax",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "mlogloss",
        "seed": 42
    }
    final_params.update(study.best_params)
    
    final_model = xgb.train(
        final_params,
        dall_w,
        num_boost_round=300,
        verbose_eval=50
    )

    # ── Save artefacts ─────────────────────────────────────────────────────
    joblib.dump(final_model, MODEL_OUT)
    print(f"\nModel saved  -> {MODEL_OUT}")

    label_map = {c: int(le.transform([c])[0]) for c in classes}
    with open(LABEL_MAP_OUT, "w") as fh:
        json.dump(label_map, fh, indent=2)
    print(f"Label map    -> {LABEL_MAP_OUT}")

    with open(FEAT_OUT, "w") as fh:
        json.dump(bands, fh, indent=2)
    print(f"Feature list -> {FEAT_OUT}")

    print("\n[OK] Stage 4 XGBoost TUNED polymer model training complete!")


if __name__ == "__main__":
    main()
