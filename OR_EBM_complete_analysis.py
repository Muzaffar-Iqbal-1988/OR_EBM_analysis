from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

LOG = logging.getLogger("or_ebm")
SEED = 42
TARGET = "Var_Project_Success"
MAIN = [
    "Fac_Facility_Readiness",
    "Fac_People_Readiness",
    "Fac_Tech_redainess", 
    "Fac_Organiz_Readiness",
]
CONTROLS = [
    "Gender", "Airport", "Job", "Organizational_tenure",
    "Organization_type", "Airport_projects",
]
NUMERIC = MAIN + ["Organizational_tenure", "Airport_projects"]
CATEGORICAL = ["Gender", "Airport", "Job", "Organization_type"]
FEATURES = MAIN + CONTROLS
LABELS = {
    MAIN[0]: "Facility readiness", MAIN[1]: "People readiness",
    MAIN[2]: "Technology readiness", MAIN[3]: "Organisational readiness",
    TARGET: "Project success",
}
ITEM_GROUPS = OrderedDict([
    ("Facility", [f"FR{i}" for i in range(1, 8)]),
    ("People", [f"PR{i}" for i in range(1, 9)]),
    ("Technology", [f"TR{i}" for i in range(1, 16)]),
    ("Organisation", [f"OR{i}" for i in range(1, 25)]),
])
ALIASES = {
    "Fac_Tech_readiness": "Fac_Tech_redainess",
    "Fac_Tech_Readiness": "Fac_Tech_redainess",
    "Infrastructure": "Airport",
    "Infrastructure_projects": "Airport_projects",
    "Infrastructure _projects": "Airport_projects",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate OR-EBM regression metrics, diagnostics, and figures."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to the input CSV.")
    parser.add_argument("--out", default=Path("outputs"), type=Path)
    parser.add_argument("--folds", type=int, default=5, help="Outer CV folds (default 5).")
    parser.add_argument("--repeats", type=int, default=3, help="Outer CV repeats (default 3).")
    parser.add_argument("--inner-folds", type=int, default=3, help="READI hyperparameter search folds.")
    parser.add_argument("--tune-iterations", type=int, default=6, help="READI randomized-search candidates.")
    parser.add_argument("--stack-folds", type=int, default=3, help="Out-of-fold meta-features.")
    parser.add_argument("--skip-readi-stack", action="store_true", help="Skip computationally expensive READI stack.")
    parser.add_argument("--skip-autoencoder", action="store_true", help="Skip optional TensorFlow deep comparator.")
    parser.add_argument("--item-analysis", action="store_true", help="Supplementary separate item-level EBM models.")
    parser.add_argument("--strict", action="store_true", help="Require 633 rows, 80 columns, zero duplicates/missing values.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run selected benchmark models; e.g., --models MLR 'OR-EBM (full controls)'. "
                             "The four-feature EBM is configured separately.")
    parser.add_argument("--skip-explanations", action="store_true",
                        help="Evaluate benchmark models without the explanatory EBM.")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced settings: 2 CV folds, 1 repeat, and 1 tuning candidate.")
    return parser.parse_args(argv)


def save_json(value, path):
    def converter(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)
    Path(path).write_text(json.dumps(value, indent=2, default=converter), encoding="utf-8")


def get_ebm_class():
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("The interpret package is required for OR-EBM. Install requirements.txt.") from exc


def load_and_audit(path, tables, strict=False):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    original_cols = list(df.columns)
    mapping = {alias: canonical for alias, canonical in ALIASES.items()
               if alias in df.columns and canonical not in df.columns}
    if mapping:
        LOG.warning("Mapped input column aliases: %s", mapping)
        df = df.rename(columns=mapping)
    missing_cols = [col for col in [TARGET] + FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required CSV columns missing: {missing_cols}. Actual columns: {list(df.columns)}")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate CSV column names after alias mapping; resolve manually.")
    n_original = len(df)
    n_duplicates = int(df.duplicated().sum())
    n_missing_all = int(df.isna().sum().sum())
    missing_target = int(df[TARGET].isna().sum())
    items = [col for group in ITEM_GROUPS.values() for col in group if col in df]
    ps_items = [col for col in df if re.fullmatch(r"PS\d+", col, flags=re.I)]
    invalid_items = {}
    for col in items + ps_items:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & (values.isna() | ~values.between(1, 5) |
                                 (values.round(0) != values))
        if int(bad.sum()):
            invalid_items[col] = int(bad.sum())
    for name in [TARGET] + NUMERIC:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    if not np.isfinite(df[TARGET].dropna().to_numpy(dtype=float)).all():
        raise ValueError("Outcome contains infinity. Verify the input dataset.")
    cleaned = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    audit = {
        "original_rows": n_original, "original_columns": len(original_cols),
        "original_cells": int(df.shape[0] * df.shape[1]),
        "missing_cells_all_columns": n_missing_all,
        "missing_model_columns": df[[TARGET] + FEATURES].isna().sum().to_dict(),
        "duplicate_complete_rows": n_duplicates,
        "missing_outcome_rows": missing_target,
        "invalid_likert_counts": invalid_items,
        "post_screening_rows": len(cleaned),
        "excluded_total": n_original - len(cleaned),
        "alias_mapping": mapping,
            }
    save_json(audit, tables / "data_quality_audit.json")
    pd.DataFrame({
        "column": df.columns, "missing": df.isna().sum().to_numpy(),
        "dtype": df.dtypes.astype(str).to_numpy(),
    }).to_csv(tables / "all_column_missingness.csv", index=False)
    if len(df) != 633 or df.shape[1] != 80:
        LOG.warning("Input shape: %s rows x %s columns; expected 633 x 80 in strict mode.", *df.shape)
    if n_missing_all or n_duplicates or invalid_items:
        LOG.warning("Data QA found missing=%d, duplicated=%d, invalid Likert=%s.",
                    n_missing_all, n_duplicates, invalid_items)
    if strict and (df.shape != (633, 80) or n_missing_all or n_duplicates or invalid_items):
        raise ValueError("--strict: input does not meet the configured shape and quality checks. See data_quality_audit.json")
    if cleaned.shape[0] < 25:
        raise ValueError("Too few observations after target/duplicate screening.")
    if cleaned[TARGET].le(0).any():
        LOG.warning("Nonpositive outcome values: MAPE will be undefined. See metrics notes.")
    # Use the four readiness variables and designated control variables as inputs.
    assert not any(re.fullmatch(r"PS\d+", x, flags=re.I) for x in FEATURES)
    assert "Var_Oper_Readiness" not in FEATURES
    return cleaned, audit


def metrics(y_true, y_pred, nominal_p=None):
    """Return regression metrics; MAPE is stored as a fraction."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if not np.isfinite(y_pred).all():
        raise ValueError("Model produced nonfinite predictions.")
    n = len(y_true)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true-y_pred)/y_true))) if np.all(y_true != 0) else np.nan
    # Nominal adjusted R² counts raw input variables. It is a descriptive
    # diagnostic, not an effective-degrees-of-freedom adjustment for nonlinear models.
    adj = (1 - (1-r2)*(n-1)/(n-nominal_p-1)
           if nominal_p is not None and n > nominal_p + 1 else np.nan)
    return {
        "N": n, "MAE": mae, "RMSE": rmse, "R2": r2,
        "Adjusted_R2_nominal": float(adj),
        "MAPE_fraction": mape,
        "MAPE_percent": float(100*mape) if np.isfinite(mape) else np.nan,
        "MAPE_complement_percent": (
            float(100*(1-mape)) if np.isfinite(mape) else np.nan),
    }


def onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def preprocess():
    """Each fit() learns imputation, numeric scaling and categories from its train fold only."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", onehot())])
    return ColumnTransformer([
        ("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)
    ], remainder="drop", sparse_threshold=0)


def wrap(estimator):
    return Pipeline([("preprocess", preprocess()), ("model", estimator)])


def ready_base(estimator):
    """Preprocessing+LASSO refit INSIDE every stacking meta-feature fold."""
    return Pipeline([
        ("preprocess", preprocess()),
        ("lasso", SelectFromModel(Lasso(alpha=0.001, max_iter=10000,
                                        random_state=SEED), threshold="median")),
        ("model", estimator),
    ])


def readi_stack(args):
    """Create a stacking regressor with four base learners and a Ridge meta-learner.

    The hyperparameter search operates on development folds. Stacking uses
    out-of-fold predictions with preprocessing and feature selection fitted
    within each corresponding training fold.
    """
    stack = StackingRegressor(
        estimators=[
            ("ridge", ready_base(Ridge(alpha=1.0))),
            ("svr", ready_base(SVR(kernel="rbf", C=10.0, epsilon=0.1))),
            ("gbr", ready_base(GradientBoostingRegressor(n_estimators=150,
                                                         learning_rate=0.05,
                                                         max_depth=2,
                                                         random_state=SEED))),
            ("bayesian", ready_base(BayesianRidge())),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=args.stack_folds, shuffle=True, random_state=SEED),
        n_jobs=1,
        passthrough=False,
    )
    params = {
        "ridge__model__alpha": [0.1, 1.0, 10.0],
        "svr__model__C": [0.1, 1.0, 10.0],
        "svr__model__gamma": ["scale", "auto"],
        "gbr__model__n_estimators": [100, 200],
        "gbr__model__learning_rate": [0.03, 0.1],
        "final_estimator__alpha": [0.1, 1.0, 10.0],
    }
    return RandomizedSearchCV(
        estimator=stack, param_distributions=params,
        n_iter=args.tune_iterations,
        cv=KFold(n_splits=args.inner_folds, shuffle=True, random_state=SEED+1),
        scoring="neg_root_mean_squared_error", n_jobs=1,
        random_state=SEED, refit=True, error_score="raise",
    )


class AutoencoderANN(BaseEstimator, RegressorMixin):
    """Autoencoder feature extraction followed by neural-network regression.

    The estimator is compatible with scikit-learn cross-validation; each fit
    trains the representation and regression network on its training fold.
    TensorFlow is required to fit this model.
    """
    def __init__(self, seed=SEED, latent_dim=8, epochs=300, batch_size=32):
        self.seed = seed
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("Autoencoder_ANN requires TensorFlow. Install requirements-full.txt or use --skip-autoencoder.") from exc
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        x = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32).ravel()
        self.y_mean_ = float(targets.mean())
        self.y_std_ = float(targets.std()) or 1.0
        ys = (targets - self.y_mean_) / self.y_std_
        latent_dim = min(self.latent_dim, x.shape[1])
        inputs = tf.keras.Input(shape=(x.shape[1],))
        h = tf.keras.layers.Dense(32, activation="relu")(inputs)
        z = tf.keras.layers.Dense(latent_dim, activation="relu")(h)
        h = tf.keras.layers.Dense(32, activation="relu")(z)
        outputs = tf.keras.layers.Dense(x.shape[1], activation="linear")(h)
        autoencoder = tf.keras.Model(inputs, outputs)
        self.encoder_ = tf.keras.Model(inputs, z)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        early_ae = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True)
        autoencoder.fit(x, x, epochs=self.epochs, batch_size=self.batch_size,
                        validation_split=0.2, callbacks=[early_ae], verbose=0)
        train_z = self.encoder_.predict(x, verbose=0)
        self.network_ = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.10), tf.keras.layers.Dense(1),
        ])
        self.network_.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        early_ann = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True)
        self.network_.fit(train_z, ys, epochs=self.epochs, batch_size=self.batch_size,
                          validation_split=0.2, callbacks=[early_ann], verbose=0)
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=np.float32)
        z = self.encoder_.predict(x, verbose=0)
        ys = self.network_.predict(z, verbose=0).reshape(-1)
        return ys * self.y_std_ + self.y_mean_


def benchmark_models(args):
    from xgboost import XGBRegressor
    EBM = None
    if args.models is None or "OR-EBM (full controls)" in args.models:
        EBM = get_ebm_class()
    base_rf = wrap(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))
    base_xgb = wrap(XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                 subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", random_state=SEED, n_jobs=1))
    conventional_stack = StackingRegressor(
        estimators=[("rf", base_rf), ("xgb", base_xgb),
                    ("svr", wrap(SVR(kernel="rbf", C=10, epsilon=0.1))),
                    ("knn", wrap(KNeighborsRegressor(n_neighbors=7, weights="distance")))],
        final_estimator=RidgeCV(),
        cv=KFold(n_splits=args.stack_folds, shuffle=True, random_state=SEED),
        n_jobs=1,
    )
    all_models = OrderedDict([
        ("MLR", wrap(LinearRegression())),
        ("Polynomial Regression", wrap(Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", RidgeCV()),
        ]))),
        ("SVR", wrap(SVR(kernel="rbf", C=10, epsilon=0.1))),
        ("KNN", wrap(KNeighborsRegressor(n_neighbors=7, weights="distance"))),
        ("Random Forest", wrap(RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1))),
        ("XGBoost", wrap(XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                      random_state=SEED, n_jobs=1))),
        ("MLP", wrap(MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                  solver="adam", alpha=0.001, learning_rate_init=0.001,
                                  max_iter=1000, early_stopping=True, random_state=SEED))),
        ("GPR", wrap(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0)*RBF(length_scale=1.0)+WhiteKernel(noise_level=1.0),
            normalize_y=True, random_state=SEED))),
        ("Stacking Ensemble Regression", conventional_stack),
        ("READI-Stack", readi_stack(args)),
        ("Autoencoder_ANN", wrap(AutoencoderANN(epochs=(20 if args.quick else 300)))),
    ])
    if EBM is not None:
        all_models["OR-EBM (full controls)"] = wrap(EBM(
            interactions=10, random_state=SEED, n_jobs=1))
    if args.skip_readi_stack:
        all_models.pop("READI-Stack", None)
    if args.skip_autoencoder:
        all_models.pop("Autoencoder_ANN", None)
    if args.models:
        unknown = set(args.models) - set(all_models)
        if unknown:
            raise ValueError(f"Unknown or skipped model names {sorted(unknown)}; available {list(all_models)}")
        all_models = OrderedDict((k, all_models[k]) for k in args.models)
    return all_models


def fit_validate(name, estimator, x_dev, y_dev, x_test, y_test, cv, p, tables):
    """CV on DEV only; final fit DEV only; untouched TEST evaluated once."""
    folds = []
    for fold, (i_train, i_valid) in enumerate(cv.split(x_dev), start=1):
        model = clone(estimator)
        model.fit(x_dev.iloc[i_train], y_dev.iloc[i_train])
        pred = model.predict(x_dev.iloc[i_valid])
        folds.append({"Model": name, "Fold": fold, **metrics(y_dev.iloc[i_valid], pred, p)})
    fold_df = pd.DataFrame(folds)
    summary = {"Model": name, "CV_folds": len(folds)}
    for key in ["MAE", "RMSE", "R2", "Adjusted_R2_nominal",
                "MAPE_fraction", "MAPE_percent",
                "MAPE_complement_percent"]:
        summary[key + "_CV"] = float(fold_df[key].mean())
        summary[key + "_CV_SD"] = float(fold_df[key].std(ddof=1)) if len(fold_df) > 1 else np.nan
    final = clone(estimator)
    final.fit(x_dev, y_dev)
    test_pred = np.asarray(final.predict(x_test), dtype=float)
    hold = {"Model": name, **{k+"_Holdout": v for k, v in metrics(y_test, test_pred, p).items()}}
    prediction_table = pd.DataFrame({"original_row_index": x_test.index,
                                     "observed": y_test.to_numpy(), "predicted": test_pred})
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    prediction_table.to_csv(tables / f"holdout_predictions_{safe}.csv", index=False)
    return summary, hold, fold_df, final


def descriptive(df, tables):
    frame = df[MAIN+[TARGET]].agg(["mean", "std", "min", "max", "skew"]).T
    frame.columns = ["Mean", "SD", "Minimum", "Maximum", "Skewness"]
    frame.index = [LABELS.get(x, x) for x in frame.index]
    frame.index.name = "Variable"
    frame.to_csv(tables / "table_5_descriptive.csv")
    corr = df[MAIN+[TARGET]].corr()
    corr.columns = corr.index = [LABELS.get(x, x) for x in corr.columns]
    corr.to_csv(tables / "figure_3_correlation_values.csv")
    return corr


def plot_correlation(corr, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(corr.to_numpy(), cmap="viridis", vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.5f}", ha="center", va="center",
                    color=("black" if val > .45 else "white"), fontsize=9)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.set_title("Correlation Matrix for Readiness Dimensions and Project Success")
    fig.colorbar(img, ax=ax, label="Pearson correlation coefficient", shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_radar(cv_table, path):
    """Top five by DEVELOPMENT CV R²; normalize relative to ALL available models."""
    if len(cv_table) < 3:
        LOG.info("Radar skipped: fewer than three available benchmark models.")
        return
    cols = ["R2_CV", "Adjusted_R2_nominal_CV",
            "MAE_CV", "RMSE_CV", "MAPE_fraction_CV"]
    if cv_table[cols].isna().any().any():
        LOG.warning("Radar skipped: missing/nonfinite metric values.")
        return
    values = cv_table[cols].copy().astype(float)
    scaled = pd.DataFrame(index=cv_table.index)
    for key in cols:
        v = values[key]
        span = v.max()-v.min()
        if span < 1e-12:
            scaled[key] = 1.0
        elif key in ("MAE_CV", "RMSE_CV", "MAPE_fraction_CV"):
            scaled[key] = (v.max()-v)/span
        else:
            scaled[key] = (v-v.min())/span
    best = cv_table.sort_values("R2_CV", ascending=False).head(5)
    angles = np.linspace(0, 2*np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    labels = ["R² CV", "Nominal adj. R²*", "Low MAE", "Low RMSE", "Low MAPE"]
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"polar": True})
    for idx in best.index:
        y = scaled.loc[idx, cols].tolist()
        y += y[:1]
        ax.plot(angles, y, lw=2, label=cv_table.loc[idx, "Model"])
        ax.fill(angles, y, alpha=.035)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 1)
    ax.set_title("CV models: normalized error and fit (0–1)", pad=25)
    ax.legend(bbox_to_anchor=(1.15, 1.12), loc="upper left", fontsize=8)
    fig.text(.5, .01, "*Nominal adjusted R² uses raw column count and is not an effective-DF correction.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .04, .86, 1])
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    scaled.insert(0, "Model", cv_table["Model"].to_numpy())
    scaled.to_csv(path.with_name("figure_4_radar_normalized_values.csv"), index=False)


def plot_validation(cv_table, holdout, path):
    merged = cv_table[["Model", "RMSE_CV"]].merge(
        holdout[["Model", "RMSE_Holdout"]], on="Model")
    merged = merged.sort_values("RMSE_CV")
    ind = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(max(10, len(merged)*.8), 6))
    ax.bar(ind-.18, merged.RMSE_CV, width=.36, label="Development cross-validation")
    ax.bar(ind+.18, merged.RMSE_Holdout, width=.36, label="Independent holdout")
    ax.set_xticks(ind, merged.Model, rotation=45, ha="right")
    ax.set_ylabel("RMSE (project-success score units)")
    ax.set_title("Development CV and independent holdout: comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def ebm_global_terms(ebm):
    explanation = ebm.explain_global()
    overall = explanation.data()
    names = list(overall["names"])
    scores = np.asarray(overall["scores"], dtype=float)
    if len(names) != len(scores):
        raise ValueError("InterpretML global explanation has mismatched term names and importances.")
    term_features = list(ebm.term_features_)
    if len(names) != len(term_features):
        LOG.warning("Global explanation names and EBM term_features_ differ in length.")
    records = []
    for i, (name, importance) in enumerate(zip(names, scores)):
        indices = tuple(int(v) for v in term_features[i]) if i < len(term_features) else ()
        records.append({"Term_index": i, "Term": name,
                        "Importance_mean_absolute_score": float(importance),
                        "Term_type": "Main" if len(indices) == 1 else
                                     "Interaction" if len(indices) == 2 else "Other",
                        "Input_columns": " & ".join(MAIN[j] for j in indices),
                        "Feature_indices": ",".join(map(str, indices))})
    return explanation, pd.DataFrame(records)


def numeric_curve(exp, term_index):
    data = exp.data(term_index)
    x = np.asarray(data["names"], dtype=float)
    y = np.asarray(data["scores"], dtype=float)
    if y.ndim != 1:
        raise ValueError(f"Main term {term_index}: unexpected score shape {y.shape}.")
    if len(x) == len(y)+1:
        mids = (x[:-1]+x[1:])/2
        return mids, y, x
    if len(x) == len(y):
        return x, y, None
    raise ValueError(f"Cannot align EBM bin names ({len(x)}) and scores ({len(y)}).")


def compute_threshold(x, contribution, frac=0.10, persistent_intervals=3):
    """Identify the largest adjacent increase and subsequent curve flattening.

    The input x values are sorted readiness scores or bin centers. Saturation
    starts at the first of three consecutive post-threshold intervals with
    changes no larger than frac times the maximum positive increase.
    Differences across uneven bins represent score jumps rather than slopes.
    """
    x = np.asarray(x, float)
    y = np.asarray(contribution, float)
    if len(x) != len(y) or len(x) < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("Threshold requires >=2 finite sorted, distinct x centers and scores.")
    d = np.diff(y)
    max_idx = int(np.argmax(d))
    largest = float(d[max_idx])
    threshold = float(x[max_idx+1]) if largest > 0 else np.nan
    sat = np.nan
    sat_start = None
    if largest > 0:
        cutoff = frac*largest
        # An interval j goes from x[j] to x[j+1]. First strictly post-
        # threshold interval starts at j=max_idx+1, not at the curve's origin.
        for j in range(max_idx+1, len(d)-persistent_intervals+1):
            if np.all(np.abs(d[j:j+persistent_intervals]) <= cutoff):
                sat = float(x[j])
                sat_start = j
                break
    return {"threshold_value": threshold, "max_positive_adjacent_jump": largest,
            "zero_contribution_crossing": interpolated_zero_crossing(x, y),
            "saturation_value": sat, "saturation_interval_start_index": sat_start,
            "saturation_rule": f"{persistent_intervals} intervals |delta| <= {frac:.2f} * largest positive jump, strictly after threshold",
            "curve_note": "Fitted contribution function; inspect bin occupancy and stability."}


def interpolated_zero_crossing(x, y):
    for i in range(1, len(y)):
        if y[i-1] < 0 <= y[i]:
            return float(x[i-1]+(0-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1]))
    return np.nan


def curves_and_thresholds(model, x_dev, figures, tables):
    exp, term_table = ebm_global_terms(model)
    term_table.sort_values("Importance_mean_absolute_score", ascending=False).to_csv(
        tables / "table_10_global_term_importance.csv", index=False)
    main_rows = term_table[term_table.Term_type == "Main"].copy()
    main_rows.to_csv(tables / "table_11_main_effect_importance.csv", index=False)
    inter_rows = term_table[term_table.Term_type == "Interaction"].copy()
    inter_rows.to_csv(tables / "table_13_interaction_importance.csv", index=False)
    plot_importance(term_table, figures / "figure_5_global_term_importance.png")
    thresholds = []
    for feature_idx, feature in enumerate(MAIN):
        matches = term_table[(term_table.Term_type == "Main") &
                             (term_table.Feature_indices == str(feature_idx))]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous EBM main term: {feature}")
        index = int(matches.iloc[0].Term_index)
        x, s, boundaries = numeric_curve(exp, index)
        if not (np.isfinite(x).all() and np.isfinite(s).all()):
            raise ValueError(f"Nonfinite curve coordinates for {feature}.")
        threshold = compute_threshold(x, s)
        # Occupancy of EBM bins in DEV training data (not holdout) to flag sparse tails.
        if boundaries is not None:
            occupancy = np.histogram(x_dev[feature].dropna().to_numpy(), bins=boundaries)[0]
        else:
            edges = np.r_[-np.inf, (x[:-1]+x[1:])/2, np.inf]
            occupancy = np.histogram(x_dev[feature].dropna().to_numpy(), bins=edges)[0]
        pd.DataFrame({"readiness_score": x, "EBM_additive_contribution": s,
                      "development_bin_n": occupancy}).to_csv(
                          tables / f"figure_{6+feature_idx}_curve_{feature}.csv", index=False)
        thresholds.append({"Feature": feature, **threshold,
                           "minimum_training_bin_n": int(np.min(occupancy)),
                           "n_training_bins_under_5": int(np.sum(occupancy < 5))})
        plot_curve(feature, x, s, threshold, occupancy,
                   figures / f"figure_{6+feature_idx}_curve_{feature}.png")
    pd.DataFrame(thresholds).to_csv(tables / "table_12_threshold_saturation.csv", index=False)
    plot_interactions(model, exp, inter_rows, x_dev, figures, tables)
    return term_table


def plot_importance(table, path):
    d = table.sort_values("Importance_mean_absolute_score").tail(14)
    fig, ax = plt.subplots(figsize=(11, max(5, .45*len(d)+1.6)))
    ax.barh(d.Term, d.Importance_mean_absolute_score)
    ax.set_xlabel("Mean absolute EBM term contribution (project-success score units)")
    ax.set_title("OR-EBM global term importance · development-trained readiness-only model")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_curve(feature, x, scores, summary, counts, path):
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(x, scores, marker="o", markersize=3, lw=1.8)
    ax.axhline(0, linestyle="--", lw=1, color="gray")
    if np.isfinite(summary["threshold_value"]):
        ax.axvline(summary["threshold_value"], linestyle=":", color="black",
                   label="Largest adjacent positive jump")
    if np.isfinite(summary["saturation_value"]):
        ax.axvline(summary["saturation_value"], linestyle="-.", color="gray",
                   label="Saturation rule")
    ax.set_xlabel(f"{LABELS[feature]} (original dataset score units)")
    ax.set_ylabel("EBM additive contribution to predicted project success")
    ax.set_title(f"Nonlinear OR-EBM contribution: {LABELS[feature]}")
    ax.grid(alpha=.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.text(.5, .005, "Fitted model contribution relative to baseline; inspect support across readiness levels.",
             fontsize=8, ha="center")
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_interactions(model, exp, interaction_table, x_dev, figures, tables):
    """Plot signed two-dimensional interaction terms from fitted EBM scores.

    Use the signed surfaces to inspect conditional patterns; global interaction
    importance summarizes the magnitude of each interaction term.
    """
    if interaction_table.empty:
        LOG.warning("Model has no pairwise interactions; no interaction surfaces generated.")
        return
    pairs = interaction_table.sort_values("Importance_mean_absolute_score", ascending=False)
    if not hasattr(model, "eval_terms"):
        raise RuntimeError("Installed interpret package lacks eval_terms(); generating signed interaction surfaces requires a compatible version.")
    for _, term in pairs.iterrows():
        term_index = int(term.Term_index)
        i, j = [int(v) for v in term.Feature_indices.split(",")]
        f1, f2 = MAIN[i], MAIN[j]
        # Observed quantiles avoid extrapolation into unsupported score regions.
        a = np.unique(np.quantile(x_dev[f1], np.linspace(.05, .95, 21)))
        b = np.unique(np.quantile(x_dev[f2], np.linspace(.05, .95, 21)))
        xx, yy = np.meshgrid(a, b, indexing="ij")
        reference = {name: float(x_dev[name].median()) for name in MAIN}
        grid = pd.DataFrame([reference]*xx.size, columns=MAIN)
        grid[f1] = xx.ravel()
        grid[f2] = yy.ravel()
        all_terms = np.asarray(model.eval_terms(grid))
        if all_terms.ndim != 2 or all_terms.shape[0] != len(grid):
            raise RuntimeError(f"Unexpected eval_terms array shape {all_terms.shape}")
        score = all_terms[:, term_index].reshape(xx.shape)
        if not np.isfinite(score).all():
            raise RuntimeError(f"Non-finite interaction values for {f1} x {f2}")
        table = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel(),
                              "signed_EBM_interaction_contribution": score.ravel()})
        prefix = f"interaction_{i+1}_{j+1}"
        table.to_csv(tables / f"{prefix}_surface.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        norm_lim = float(max(abs(score.min()), abs(score.max()), 1e-9))
        image = ax.pcolormesh(a, b, score.T, shading="auto", cmap="RdBu_r",
                              vmin=-norm_lim, vmax=norm_lim)
        fig.colorbar(image, ax=ax, label="Signed interaction contribution to prediction")
        ax.set_xlabel(LABELS[f1]); ax.set_ylabel(LABELS[f2])
        ax.set_title(f"OR-EBM pairwise interaction: {LABELS[f1]} × {LABELS[f2]}")
        fig.tight_layout()
        fig.savefig(figures / f"{prefix}_heatmap.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
        # Conditional interaction contrast describes fitted model behavior.
        if len(b) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(a, score[:, 0], label=f"{LABELS[f2]} = {b[0]:.2f} (5th percentile)")
            ax.plot(a, score[:, -1], label=f"{LABELS[f2]} = {b[-1]:.2f} (95th percentile)")
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_xlabel(LABELS[f1]); ax.set_ylabel("Pairwise component contribution")
            ax.set_title(f"Conditional pairwise component: {LABELS[f1]}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures / f"{prefix}_conditional_lines.png", dpi=250, bbox_inches="tight")
            plt.close(fig)


def plot_pipeline(path, dev_n, holdout_n):
    stages = [
        ("Dataset acquisition + QA", "Original survey n=724; analytic CSV selection MUST be documented"),
        ("Holdout isolation", f"Development n={dev_n} · held-out test n={holdout_n}"),
        ("Train-fold preprocessing", "Imputation contingency · scaling · categorical one-hot encoding"),
        ("Benchmark comparison", "MLR, polynomial, SVR, KNN, RF, XGB, MLP, GPR, stacking, READI, AE-ANN"),
        ("Separate OR-EBM specifications", "Readiness only = interpretation; full controls = sensitivity / benchmark"),
        ("Evaluation", "Development CV + independent holdout · MAE, RMSE, R², MAPE"),
        ("Explanations", "Feature importance · curves · threshold/saturation · signed interactions"),
    ]
    fig, ax = plt.subplots(figsize=(11.7, 12.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13.6); ax.axis("off")
    for i, (header, subtitle) in enumerate(stages):
        y = 12.25-i*1.87
        patch = FancyBboxPatch((.6, y), 8.8, 1.35, boxstyle="round,pad=0.12",
                               fc="white", ec="black", lw=1.4)
        ax.add_patch(patch)
        ax.text(5, y+.88, header, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(5, y+.36, subtitle, ha="center", va="center", fontsize=8.5)
        if i < len(stages)-1:
            ax.add_patch(FancyArrowPatch((5, y-.12), (5, y-.48), arrowstyle="-|>",
                                         mutation_scale=14, color="black", lw=1.2))
    ax.set_title("OR-EBM workflow: independent validation and explanatory analysis", fontsize=14, pad=14)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def supplementary_items(dev_df, y_dev, figures, tables):
    EBM = get_ebm_class()
    rows = []
    for group, possible in ITEM_GROUPS.items():
        cols = [col for col in possible if col in dev_df.columns]
        if len(cols) < 2:
            LOG.warning("Skipping item model %s (only %s item columns present).", group, len(cols))
            continue
        model = EBM(interactions=min(5, len(cols)*(len(cols)-1)//2),
                    random_state=SEED, n_jobs=1)
        x = dev_df[cols].apply(pd.to_numeric, errors="coerce")
        model.fit(x, y_dev)
        exp = model.explain_global()
        for idx, (indices, importance) in enumerate(zip(model.term_features_, exp.data()["scores"])):
            if len(indices) == 1:
                rows.append({"Dimension": group, "Item": cols[indices[0]],
                             "Importance_mean_absolute_score": float(importance),
                             "Analysis_type": "Separate item-level model"})
    if rows:
        result = pd.DataFrame(rows).sort_values(["Dimension", "Importance_mean_absolute_score"],
                                                ascending=[True, False])
        result.to_csv(tables / "supplementary_item_importance.csv", index=False)
        result.groupby("Dimension", as_index=False).head(1).to_csv(
            tables / "supplementary_top_item_per_dimension.csv", index=False)


def record_environment(tables, args):
    import sklearn
    import xgboost
    package_versions = {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "xgboost": xgboost.__version__, "random_seed": SEED,
                        "cli_config": vars(args)}
    for package in ("interpret", "tensorflow"):
        try:
            import importlib.metadata
            package_versions[package] = importlib.metadata.version(package)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            package_versions[package] = "not installed"
    save_json(package_versions, tables / "environment.json")


def main(argv=None):
    args = arguments(argv)
    if args.quick:
        args.folds = 2; args.repeats = 1; args.inner_folds = 2
        args.tune_iterations = 1; args.stack_folds = 2
        LOG.warning("Quick mode uses reduced cross-validation and tuning settings.")
    if args.folds < 2 or args.repeats < 1 or args.inner_folds < 2 or args.stack_folds < 2 or args.tune_iterations < 1:
        raise ValueError("Invalid fold, repeat or tuning settings.")
    root = args.out
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(root / "run.log", encoding="utf-8")])
    random.seed(SEED); np.random.seed(SEED)
    record_environment(tables, args)
    data, qa = load_and_audit(args.data, tables, strict=args.strict)
    correlation = descriptive(data, tables)
    plot_correlation(correlation, figures / "figure_3_correlation_matrix.png")

    # Partition the dataset before cross-validation and model tuning.
    x = data[FEATURES].copy()
    y = data[TARGET].astype(float).copy()
    x_dev, x_hold, y_dev, y_hold = train_test_split(
        x, y, test_size=.30, random_state=SEED, shuffle=True)
    assert x_dev.index.intersection(x_hold.index).empty
    indices = pd.DataFrame({"original_row_index": data.index,
                            "partition": ["development" if idx in x_dev.index else "holdout"
                                          for idx in data.index]})
    indices.to_csv(tables / "data_partition_indices.csv", index=False)
    LOG.info("Development n=%d; isolated holdout n=%d", len(x_dev), len(x_hold))
    if args.strict and (len(x_dev), len(x_hold)) != (443, 190):
        raise ValueError("Expected original 633-row 70:30 split of 443/190.")
    plot_pipeline(figures / "figure_2_modelling_flowchart.png", len(x_dev), len(x_hold))
    cv = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    models = benchmark_models(args)
    cv_rows, hold_rows, fold_rows = [], [], []
    for name, estimator in models.items():
        LOG.info("Development CV and holdout: %s", name)
        cv_summary, hold_summary, all_folds, fitted = fit_validate(
            name, estimator, x_dev, y_dev, x_hold, y_hold, cv, len(FEATURES), tables)
        cv_rows.append(cv_summary); hold_rows.append(hold_summary); fold_rows.append(all_folds)
        if isinstance(fitted, RandomizedSearchCV):
            save_json({"model": name, "best_hyperparameters": fitted.best_params_,
                       "inner_development_RMSE": -float(fitted.best_score_)},
                      tables / "readi_final_development_tuning.json")
        LOG.info("%s: CV R² %.4f RMSE %.4f | holdout R² %.4f RMSE %.4f",
                 name, cv_summary["R2_CV"], cv_summary["RMSE_CV"],
                 hold_summary["R2_Holdout"], hold_summary["RMSE_Holdout"])

    if not args.skip_explanations:
        EBM = get_ebm_class()
        name = "OR-EBM (readiness only)"
        estimator = EBM(interactions=6, random_state=SEED, n_jobs=1)
        LOG.info("Main explanatory four-readiness OR-EBM, CV + held-out evaluation")
        cv_sum, hold_sum, folds, trained = fit_validate(
            name, estimator, x_dev[MAIN], y_dev, x_hold[MAIN], y_hold, cv, len(MAIN), tables)
        cv_rows.append(cv_sum); hold_rows.append(hold_sum); fold_rows.append(folds)
        # The 70:30 split above validates predictive performance of the
        # readiness-only OR-EBM. The interpretation outputs (contribution curves,
        # thresholds, saturation points, global importance and interactions),
        # however, are intended to describe the fitted relationships across the
        # whole analytical sample, so a dedicated interpretation model is fitted
        # on all rows and used for curve and threshold/saturation extraction.
        interpretation = EBM(interactions=6, random_state=SEED, n_jobs=1)
        interpretation.fit(x[MAIN], y)
        curves_and_thresholds(interpretation, x[MAIN], figures, tables)
        if args.item_analysis:
            supplementary_items(data.loc[x_dev.index], y_dev, figures, tables)
    elif args.item_analysis:
        LOG.warning("--item-analysis has no effect with --skip-explanations.")

    cv_df = pd.DataFrame(cv_rows)
    hold_df = pd.DataFrame(hold_rows)
    if cv_df.empty:
        raise ValueError("No models evaluated.")
    cv_df.sort_values("R2_CV", ascending=False).to_csv(tables / "table_6_development_cv.csv", index=False)
    hold_df.sort_values("R2_Holdout", ascending=False).to_csv(
        tables / "table_7_independent_holdout.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(tables / "development_all_fold_metrics.csv", index=False)
    stability = cv_df[["Model", "RMSE_CV", "R2_CV"]].merge(
        hold_df[["Model", "RMSE_Holdout", "R2_Holdout"]], on="Model")
    stability["Delta_RMSE_Holdout_minus_CV"] = stability.RMSE_Holdout - stability.RMSE_CV
    stability["Delta_R2_Holdout_minus_CV"] = stability.R2_Holdout - stability.R2_CV
    stability.to_csv(tables / "table_8_generalization_difference.csv", index=False)
    merged = cv_df.merge(hold_df, on="Model")
    merged.to_csv(tables / "all_metrics_cv_and_holdout.csv", index=False)
    metric_summary = []
    for population, frame, suffix in [("Development CV", cv_df, "_CV"),
                                       ("Holdout", hold_df, "_Holdout")]:
        for metric in ("MAE", "RMSE", "R2", "MAPE_fraction"):
            col = metric+suffix
            if frame[col].notna().any():
                row = frame.loc[frame[col].idxmax() if metric == "R2" else frame[col].idxmin()]
                metric_summary.append({"Population": population, "Metric": metric,
                                       "Lowest_error_or_largest_R2_model": row.Model,
                                       "Value": float(row[col])})
    pd.DataFrame(metric_summary).to_csv(tables / "table_9_metric_specific_comparisons.csv", index=False)
    # Construct the five-model radar using the benchmark comparison.
    benchmark_cv = cv_df[cv_df.Model != "OR-EBM (readiness only)"].copy()
    plot_radar(benchmark_cv, figures / "figure_4_radar_cv.png")
    plot_validation(cv_df, hold_df, figures / "cv_vs_holdout_rmse.png")
    LOG.info("Finished. Output: %s; tables=%s; figures=%s", root, tables, figures)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Pipeline failed. Examine the traceback and generated logs.")
        raise
