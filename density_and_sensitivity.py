from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import OR_EBM_complete_analysis as core 

MAIN, TARGET, SEED = core.MAIN, core.TARGET, core.SEED
SHORT = {MAIN[0]: "Facility", MAIN[1]: "People",
         MAIN[2]: "Technology", MAIN[3]: "Organisation"}
CUTOFFS = (0.05, 0.10, 0.15)


def load(data_path):
    """Load the analytical CSV with the same alias mapping and screening as the
    main pipeline (duplicate rows dropped; rows with a missing target dropped)."""
    df = pd.read_csv(data_path)
    mapping = {a: c for a, c in core.ALIASES.items()
               if a in df.columns and c not in df.columns}
    if mapping:
        df = df.rename(columns=mapping)
    missing = [c for c in MAIN + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    for c in MAIN + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fit_curves(df):
    """Fit the readiness-only interpretation OR-EBM on the full sample and return
    the (bin-centre, contribution) curve for each main effect."""
    x = df[MAIN]
    y = df[TARGET].astype(float)
    model = core.get_ebm_class()(interactions=6, random_state=SEED, n_jobs=1)
    model.fit(x, y)
    exp, term_table = core.ebm_global_terms(model)
    curves = {}
    for j, feature in enumerate(MAIN):
        ti = [i for i, tf in enumerate(model.term_features_) if tf == (j,)][0]
        xc, s, _ = core.numeric_curve(exp, ti)
        curves[feature] = (xc, s)
    return curves


def sensitivity_table(curves):
    """Re-apply the saturation rule at each cut-off; threshold rule unchanged."""
    rows = []
    for feature in MAIN:
        xc, s = curves[feature]
        base = core.compute_threshold(xc, s, 0.10)
        row = {"Readiness_dimension": SHORT[feature],
               "Estimated_threshold": base["threshold_value"],
               "Largest_positive_jump": round(base["max_positive_adjacent_jump"], 4)}
        for c in CUTOFFS:
            sv = core.compute_threshold(xc, s, c)["saturation_value"]
            row[f"Saturation_cutoff_{c:.2f}"] = (
                "Not detected" if not np.isfinite(sv) else round(float(sv), 2))
        rows.append(row)
    return pd.DataFrame(rows)


def density_support_table(df, curves, halfwidths=(2.5, 5.0)):
    """Locate each threshold/saturation point within its factor-score
    distribution and count the surrounding observations."""
    rows = []
    for feature in MAIN:
        xc, s = curves[feature]
        r = core.compute_threshold(xc, s, 0.10)
        v = df[feature].dropna().to_numpy(float)
        for kind, pt in (("threshold", r["threshold_value"]),
                         ("saturation", r["saturation_value"])):
            if not np.isfinite(pt):
                continue
            rec = {"Dimension": SHORT[feature], "Point": kind,
                   "Value": round(float(pt), 2),
                   "score_min": int(v.min()), "score_max": int(v.max()),
                   "median": float(np.median(v))}
            for h in halfwidths:
                rec[f"n_within_{h:g}"] = int(np.sum(np.abs(v - pt) <= h))
            rec["pct_at_or_below"] = round(100 * float(np.mean(v <= pt)), 1)
            rec["nearest_obs_distance"] = round(float(np.min(np.abs(v - pt))), 2)
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_density(df, curves, path):
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax, feature in zip(axes.ravel(), MAIN):
        xc, s = curves[feature]
        r = core.compute_threshold(xc, s, 0.10)
        v = df[feature].dropna().to_numpy(float)
        bins = np.arange(v.min(), v.max() + 2) - 0.5
        ax.hist(v, bins=bins, color="#c6dbef", edgecolor="#6baed6", linewidth=.4, zorder=2)
        ax2 = ax.twinx()
        xs = np.linspace(v.min(), v.max(), 400)
        ax2.plot(xs, gaussian_kde(v)(xs), color="#08519c", lw=1.6, zorder=3)
        ax2.set_yticks([])
        ymax = ax.get_ylim()[1]

        def mark(pt, color, name):
            if not np.isfinite(pt):
                return
            n25 = int(np.sum(np.abs(v - pt) <= 2.5))
            pct = 100 * float(np.mean(v <= pt))
            sparse = n25 < 10
            ax.axvline(pt, color=color, lw=2, zorder=4,
                       ls=("--" if name == "Threshold" else "-."))
            ax.annotate(f"{name} {pt:g}\nn(±2.5)={n25}\n{pct:.0f}% ≤",
                        xy=(pt, ymax * 0.98), xytext=(4, -2),
                        textcoords="offset points", fontsize=8, color=color,
                        va="top", ha="left",
                        fontweight=("bold" if sparse else "normal"))
            if sparse:
                ax.axvspan(pt - 2.5, pt + 2.5, color=color, alpha=0.08, zorder=1)

        mark(r["threshold_value"], "#d94801", "Threshold")
        mark(r["saturation_value"], "#238b45", "Saturation")
        ax.set_title(f"{SHORT[feature]} readiness (n={len(v)}; "
                     f"range {int(v.min())}–{int(v.max())}, median {int(np.median(v))})")
        ax.set_xlabel("Factor score")
        ax.set_ylabel("Number of observations")
    fig.suptitle("Factor-score distributions with OR-EBM threshold and saturation points",
                 fontsize=13.5, y=0.995)
    fig.text(0.5, 0.005, "Bold, shaded labels mark points located in sparsely populated regions "
             "(fewer than 10 observations within ±2.5 score units).",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="OR-EBM factor-score density check and saturation-cutoff sensitivity.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("outputs"), type=Path)
    args = ap.parse_args(argv)
    tables = args.out / "tables"
    figures = args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    curves = fit_curves(df)

    sens = sensitivity_table(curves)
    sens.to_csv(tables / "table_sensitivity_saturation_cutoffs.csv", index=False)

    dens = density_support_table(df, curves)
    dens.to_csv(tables / "factor_score_density_support.csv", index=False)

    plot_density(df, curves, figures / "factor_score_density_check.png")

    print("Saturation-cutoff sensitivity (threshold rule unchanged):")
    print(sens.to_string(index=False))
    print("\nFactor-score density support around each reported point:")
    print(dens.to_string(index=False))
    print(f"\nSaved tables to {tables} and the density figure to {figures}.")


if __name__ == "__main__":
    main()
