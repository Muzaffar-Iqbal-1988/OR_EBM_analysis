from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd

import OR_EBM_complete_analysis as core  # reuse the study's own functions

MAIN, TARGET, SEED = core.MAIN, core.TARGET, core.SEED

# module-level state for worker processes
_X = _Y = _GRID = _MED = None


def _load(data_path):
    df = pd.read_csv(data_path)
    mapping = {a: c for a, c in core.ALIASES.items() if a in df.columns and c not in df.columns}
    if mapping:
        df = df.rename(columns=mapping)
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    for c in MAIN + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[MAIN].reset_index(drop=True)
    Y = df[TARGET].astype(float).reset_index(drop=True)
    return X, Y


def _init(data_path):
    global _X, _Y, _GRID, _MED
    _X, _Y = _load(data_path)
    _GRID = {f: np.arange(int(_X[f].min()), int(_X[f].max()) + 1) for f in MAIN}
    _MED = {f: float(_X[f].median()) for f in MAIN}


def _fit_extract(X, Y, grid, med):
    m = core.get_ebm_class()(interactions=6, random_state=SEED, n_jobs=1)
    m.fit(X, Y)
    exp, tt = core.ebm_global_terms(m)
    main, inter = {}, {}
    for _, r in tt.iterrows():
        if r.Term_type == "Main":
            main[r.Input_columns] = float(r.Importance_mean_absolute_score)
        elif r.Term_type == "Interaction":
            inter[r.Input_columns] = float(r.Importance_mean_absolute_score)
    thr, sat, curves = {}, {}, {}
    for j, f in enumerate(MAIN):
        ti = [i for i, tf in enumerate(m.term_features_) if tf == (j,)][0]
        xc, s, _ = core.numeric_curve(exp, ti)
        r = core.compute_threshold(xc, s, 0.10)
        thr[f] = float(r["threshold_value"]); sat[f] = float(r["saturation_value"])
        g = grid[f]
        gdf = pd.DataFrame({ff: np.full(len(g), med[ff]) for ff in MAIN}); gdf[f] = g
        curves[f] = np.asarray(m.eval_terms(gdf))[:, ti].astype(float).tolist()
    return dict(main=main, inter=inter, thr=thr, sat=sat, curves=curves)


def _one(b):
    rng = np.random.RandomState(b)
    idx = rng.randint(0, len(_X), len(_X))
    return _fit_extract(_X.iloc[idx], _Y.iloc[idx], _GRID, _MED)


def vif(X):
    from sklearn.linear_model import LinearRegression
    out = {}
    for f in MAIN:
        others = [c for c in MAIN if c != f]
        r2 = LinearRegression().fit(X[others], X[f]).score(X[others], X[f])
        out[f] = float(1.0 / (1.0 - r2))
    return out


def summarise(reference, results, out_dir):
    tab = out_dir / "tables"; tab.mkdir(parents=True, exist_ok=True)
    n = len(results)
    def ci(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return np.mean(a), np.std(a, ddof=1), np.percentile(a, 2.5), np.percentile(a, 97.5)
    rows = []
    for f in MAIN:
        m, sd, lo, hi = ci([r["main"][f] for r in results])
        rows.append(dict(Term=f, Type="Main", Reference=round(reference["main"][f], 3),
                         Boot_mean=round(m, 3), SD=round(sd, 3), CI_low=round(lo, 3), CI_high=round(hi, 3)))
    for k in reference["inter"]:
        m, sd, lo, hi = ci([r["inter"].get(k, 0.0) for r in results])
        rows.append(dict(Term=k, Type="Interaction", Reference=round(reference["inter"][k], 3),
                         Boot_mean=round(m, 3), SD=round(sd, 3), CI_low=round(lo, 3), CI_high=round(hi, 3)))
    pd.DataFrame(rows).to_csv(tab / "bootstrap_importance_summary.csv", index=False)
    org = np.array([r["main"]["Fac_Organiz_Readiness"] for r in results])
    tech = np.array([r["main"]["Fac_Tech_redainess"] for r in results])
    rank1 = np.mean([max(r["main"], key=r["main"].get) == "Fac_Organiz_Readiness" for r in results])
    summary = dict(n_replications=n,
                   organisation_rank1_frac=float(rank1),
                   organisation_gt_technology_frac=float(np.mean(org > tech)),
                   corr_org_tech_importance=float(np.corrcoef(org, tech)[0, 1]))
    trows = []
    for f in MAIN:
        thr = [r["thr"][f] for r in results]
        sat = np.array([r["sat"][f] for r in results], float); det = np.isfinite(sat)
        tm, tsd, tlo, thi = ci(thr)
        srow = ci(sat[det]) if det.sum() else (np.nan, np.nan, np.nan, np.nan)
        trows.append(dict(Dimension=f, Ref_threshold=reference["thr"][f],
                          Thr_median=round(float(np.median(thr)), 2),
                          Thr_CI_low=round(tlo, 2), Thr_CI_high=round(thi, 2),
                          Sat_detected_frac=round(float(det.mean()), 3),
                          Sat_median=(None if not det.sum() else round(float(np.median(sat[det])), 2))))
    pd.DataFrame(trows).to_csv(tab / "bootstrap_threshold_saturation.csv", index=False)
    core.save_json(summary, tab / "bootstrap_ranking_summary.json")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="OR-EBM bootstrap stability analysis.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("outputs"), type=Path)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    X, Y = _load(args.data)
    grid = {f: np.arange(int(X[f].min()), int(X[f].max()) + 1) for f in MAIN}
    med = {f: float(X[f].median()) for f in MAIN}
    reference = _fit_extract(X, Y, grid, med)
    reference["grid"] = {f: grid[f].tolist() for f in MAIN}
    reference["vif"] = vif(X)

    import multiprocessing as mp
    results = []
    with mp.Pool(processes=args.workers, initializer=_init, initargs=(args.data,),
                 maxtasksperchild=20) as pool:
        for i, res in enumerate(pool.imap_unordered(_one, range(args.reps)), 1):
            results.append(res)
            if i % 25 == 0:
                print(f"{i}/{args.reps} bootstrap replications complete", flush=True)

    with open(args.out / "bootstrap_results.pkl", "wb") as f:
        pickle.dump(dict(reference=reference, results=results), f)
    summary = summarise(reference, results, args.out)
    print("VIF:", {k: round(v, 2) for k, v in reference["vif"].items()})
    print("Summary:", summary)
    print(f"Saved bootstrap results and summaries to {args.out}")


if __name__ == "__main__":
    main()
