# Operations Readiness Explainable Boosting Model (OR-EBM)

Python workflow for project-success regression, benchmark comparison, and interpretable readiness analysis.
## Dataset

Place the authorized analytical CSV at `data/Readiness.csv`. The target is `Var_Project_Success` and the four readiness variables are `Fac_Facility_Readiness`, `Fac_People_Readiness`, `Fac_Tech_redainess`, and `Fac_Organiz_Readiness`. Optional control inputs are `Gender`, `Airport`, `Job`, `Organizational_tenure`, `Organization_type`, and `Airport_projects`.

The analysis reads the four readiness score columns directly. Project-success item scores (`PS1`–`PS14`) and `Var_Oper_Readiness` are excluded from the predictor set. Input datasets are kept separately from the repository.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
```

For the classical and explainable models without TensorFlow, install `requirements.txt` instead and add `--skip-autoencoder` to the run command. Save the package versions associated with the executed analysis for consistent environments.

## Run

```bash
python OR_EBM_complete_analysis.py --data data/Readiness.csv --out outputs --strict
```

The default configuration uses a random 70:30 development/holdout split, followed by 5-fold cross-validation repeated 3 times on development data. Preprocessing, feature selection, stacking, and tuning are fitted within their training folds. Models are fitted on the development partition and evaluated on the separate holdout partition. The expected partition sizes are 443 and 190 for a 633-row input.

Other options:

```bash
# Run the core models without TensorFlow
python OR_EBM_complete_analysis.py --data data/Readiness.csv --out outputs --skip-autoencoder

# Add separate item-level analysis
python OR_EBM_complete_analysis.py --data data/Readiness.csv --out outputs --item-analysis

# Short execution for checking the software environment
python OR_EBM_complete_analysis.py --data data/Readiness.csv --out smoke --quick --skip-autoencoder --skip-readi-stack
```

`--quick`, `--models`, and `--skip-*` restrict the execution and consequently the set of generated results.

## Models

Eleven comparator configurations are available: MLR, polynomial regression, SVR, KNN, random forest, XGBoost, MLP, Gaussian process regression, conventional stacking, READI-Stack, and Autoencoder_ANN. A control-adjusted OR-EBM is included in the benchmark comparison. A distinct, four-readiness OR-EBM is fitted for its own evaluation and for interpretation.

The predictive performance of the four-readiness OR-EBM is validated on the 70:30 development/holdout split. The interpretation outputs (contribution curves, thresholds, saturation points, global importance, and interactions) are estimated from the four-readiness OR-EBM fitted on the full analytical sample, because these describe the fitted relationships across all observations rather than out-of-sample prediction.

READI-Stack uses Ridge, RBF-SVR, gradient boosting, and Bayesian Ridge base estimators; a Ridge meta-learner; and training-fold feature selection and hyperparameter search. Autoencoder_ANN uses training-fold-only representation learning and target scaling.

## Evaluation and interpretation

Outputs include MAE, RMSE, R², MAPE, nominal adjusted R², and the percentage complement of MAPE. `Adjusted_R2_nominal` uses the raw input count and is not an effective-complexity correction for nonlinear estimators. `MAPE_complement_percent` is `100 × (1 − MAPE)` and is not classification accuracy.

For each four-readiness EBM function, a threshold is defined by the largest positive change between adjacent fitted contribution scores. A saturation point is the first strictly post-threshold point satisfying the specified low-change criterion over three successive intervals. Zero-crossings are calculated separately. These are descriptive properties of the fitted functions; their values can depend on the dataset, model settings, and binning. Pairwise importance indicates interaction magnitude; signed surfaces provide additional detail about conditional patterns. These fitted associations alone do not establish causal effects.

## Stability analysis

A nonparametric bootstrap assesses how consistently the OR-EBM interpretation reproduces under resampling:

```bash
python stability_bootstrap.py --data data/Readiness.csv --out outputs --reps 500
```

For each of the `--reps` bootstrap resamples (drawn with replacement) the readiness-only OR-EBM is refitted with the identical configuration (`interactions=6`, `random_state=42`) and the feature/interaction importance, contribution curves and threshold/saturation estimates are re-extracted. Outputs include `bootstrap_importance_summary.csv` (means, SDs and 95% percentile intervals), `bootstrap_threshold_saturation.csv`, `bootstrap_ranking_summary.json` (ranking-reproduction frequencies and the Organisation-vs-Technology separation), variance inflation factors, and `bootstrap_results.pkl`.

## Diagnostics: density check and saturation-cutoff sensitivity

Two reviewer diagnostics reuse the fitted interpretation OR-EBM:

```bash
python density_and_sensitivity.py --data data/Readiness.csv --out outputs
```

This re-applies the saturation rule to the fitted contribution curves at cut-offs of 0.05, 0.10 and 0.15 (the threshold rule is unchanged), and examines the distribution of each of the four factor scores, locating the OR-EBM threshold and saturation points within each distribution and counting the surrounding observations. Outputs include `tables/table_sensitivity_saturation_cutoffs.csv`, `tables/factor_score_density_support.csv`, and `figures/factor_score_density_check.png`.

## Output files

`outputs/tables/` contains data-quality summaries, variable statistics, correlation values, partition membership, fold metrics, cross-validation and holdout comparisons, feature and interaction importance, threshold calculations, prediction tables, and runtime environment information.

`outputs/figures/` contains the modelling flowchart, correlation heatmap, normalized radar comparison, global importance chart, four readiness contribution curves, interaction plots, and validation comparisons. Tables and figures are computed from the CSV supplied at runtime.

