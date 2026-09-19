# Operations Readiness Explainable Boosting Model (OR-EBM)

Python workflow for project-success regression, benchmark comparison, and interpretable readiness analysis.
## Dataset

Place the authorized analytical CSV at `data/readiness.csv`. The target is `Var_Project_Success` and the four readiness variables are `Fac_Facility_Readiness`, `Fac_People_Readiness`, `Fac_Tech_redainess`, and `Fac_Organiz_Readiness`. Optional control inputs are `Gender`, `Airport`, `Job`, `Organizational_tenure`, `Organization_type`, and `Airport_projects`.

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
python OR_EBM_complete_analysis.py --data data/readiness.csv --out outputs --strict
```

The default configuration uses a random 70:30 development/holdout split, followed by 5-fold cross-validation repeated 3 times on development data. Preprocessing, feature selection, stacking, and tuning are fitted within their training folds. Models are fitted on the development partition and evaluated on the separate holdout partition. The expected partition sizes are 443 and 190 for a 633-row input.

Other options:

```bash
# Run the core models without TensorFlow
python OR_EBM_complete_analysis.py --data data/readiness.csv --out outputs --skip-autoencoder

# Add separate item-level analysis
python OR_EBM_complete_analysis.py --data data/readiness.csv --out outputs --item-analysis

# Short execution for checking the software environment
python OR_EBM_complete_analysis.py --data data/readiness.csv --out smoke --quick --skip-autoencoder --skip-readi-stack
```

`--quick`, `--models`, and `--skip-*` restrict the execution and consequently the set of generated results.

## Models

Eleven comparator configurations are available: MLR, polynomial regression, SVR, KNN, random forest, XGBoost, MLP, Gaussian process regression, conventional stacking, READI-Stack, and Autoencoder_ANN. A control-adjusted OR-EBM is included in the benchmark comparison. A distinct, four-readiness OR-EBM is fitted for its own evaluation and for interpretation.

READI-Stack uses Ridge, RBF-SVR, gradient boosting, and Bayesian Ridge base estimators; a Ridge meta-learner; and training-fold feature selection and hyperparameter search. Autoencoder_ANN uses training-fold-only representation learning and target scaling.

## Evaluation and interpretation

Outputs include MAE, RMSE, R², MAPE, nominal adjusted R², and the percentage complement of MAPE. `Adjusted_R2_nominal` uses the raw input count and is not an effective-complexity correction for nonlinear estimators. `MAPE_complement_percent` is `100 × (1 − MAPE)` and is not classification accuracy.

For each four-readiness EBM function, a threshold is defined by the largest positive change between adjacent fitted contribution scores. A saturation point is the first strictly post-threshold point satisfying the specified low-change criterion over three successive intervals. Zero-crossings are calculated separately. These are descriptive properties of the fitted functions; their values can depend on the dataset, model settings, and binning. Pairwise importance indicates interaction magnitude; signed surfaces provide additional detail about conditional patterns. These fitted associations alone do not establish causal effects.

## Output files

`outputs/tables/` contains data-quality summaries, variable statistics, correlation values, partition membership, fold metrics, cross-validation and holdout comparisons, feature and interaction importance, threshold calculations, prediction tables, and runtime environment information.

`outputs/figures/` contains the modelling flowchart, correlation heatmap, normalized radar comparison, global importance chart, four readiness contribution curves, interaction plots, and validation comparisons. Tables and figures are computed from the CSV supplied at runtime.

