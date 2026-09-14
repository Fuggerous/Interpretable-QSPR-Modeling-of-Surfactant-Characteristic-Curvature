# Interpretable QSPR Modeling of Surfactant Characteristic Curvature

Code and data for a quantitative structure–property relationship (QSPR) model that predicts the **characteristic curvature (Cc)** of anionic and nonionic surfactants from two-dimensional RDKit molecular descriptors, with SHAP and partial-dependence interpretation.

Cc is the surfactant term in the hydrophilic–lipophilic deviation (HLD) framework. Measuring it experimentally (salinity phase-inversion scans, interfacial tension series) is slow, so a structure-based estimate is useful for narrowing a candidate list before going to the bench.

**Scope.** 63 surfactants — 25 anionic (sulfates / sulfonates / sulfosuccinates) and 38 nonionic ethoxylates — curated from 16 literature sources. Cationic, zwitterionic, gemini and other structurally distinct classes are **not** represented, and predictions outside the represented chemistry should not be trusted.

---

## Repository contents

| Path | What it is |
|---|---|
| `Interpretable QSPR Modeling of Surfactant Characteristic Curvature.ipynb` | The full analysis: descriptor generation → filtering → target transformation → nested CV benchmarking of four tree models → SHAP and PDP interpretation. Outputs are cleared; run top to bottom to regenerate. |
| `result/Mixed_exclude.xlsx` | **Source dataset.** 63 rows × 4 columns: `SMILES`, `EO`, `PO`, `Cc`. This is the only hand-curated file — everything else is derived from it. |
| `Case4_1.xlsx` | Generated. 63 × 214 — the source columns plus all 210 RDKit 1D/2D descriptors. |
| `Case4_1_clean.xlsx` | Generated. 63 × 28 — `Cc` plus the 27 descriptors that survive filtering. |
| `requirements.txt` | Pinned dependency versions. |

`descriptors_cleaned_RDkit_mix_Exclude.xlsx` is written by the notebook and is byte-identical to `Case4_1_clean.xlsx`; it is gitignored rather than committed.

### The dataset

`result/Mixed_exclude.xlsx`:

| Column | Type | Notes |
|---|---|---|
| `SMILES` | str | Canonical SMILES. Sodium counterions were stripped from all anionic surfactants, so anionic heads appear as the bare anion (e.g. `CCCCCCCC(=O)[O-]`). |
| `EO` | int | Number of ethylene oxide units, 0–14. Bookkeeping only — dropped before modeling. |
| `PO` | int | Number of propylene oxide units, 0–18. Bookkeeping only — dropped before modeling. |
| `Cc` | float | Experimental characteristic curvature. Range −4.40 to +3.50, mean −0.95, skewness +0.56. |

---

## Environment

The notebook was run on **CPython 3.12** (kernel metadata records 3.12.3), Windows x64.

| Package | Version | Used for |
|---|---|---|
| rdkit | 2024.3.5 | SMILES parsing, sanitization, the 210 `Descriptors.descList` descriptors |
| scikit-learn | 1.4.2 | `train_test_split`, `KFold`, `GridSearchCV`, `PowerTransformer`, `RobustScaler`, `MinMaxScaler`, DT / RF / GBR, metrics, `partial_dependence` |
| xgboost | 2.1.0 | `XGBRegressor` — the final model |
| shap | 0.46.0 | `TreeExplainer`, bar and beeswarm plots |
| numpy | 2.0.1 | |
| pandas | 2.3.3 | |
| scipy | 1.13.1 | skewness / kurtosis |
| matplotlib | 3.10.7 | figures |
| seaborn | 0.13.2 | correlation heatmaps |
| openpyxl | 3.1.5 | `.xlsx` I/O (required by pandas) |
| joblib | 1.4.2 | |
| tqdm | 4.66.5 | |

The RDKit version matters: `Descriptors.descList` is not stable across releases, and 2024.3.5 is what yields the 210-descriptor starting set used here. A different RDKit will change both the descriptor count and, potentially, which 27 survive filtering.

### Setup

```bash
git clone https://github.com/Fuggerous/Interpretable-QSPR-Modeling-of-Surfactant-Characteristic-Curvature.git
cd Interpretable-QSPR-Modeling-of-Surfactant-Characteristic-Curvature

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

Open the notebook and run all cells. Paths inside it are **relative to the repository root**, so launch Jupyter from there. Total runtime is dominated by the nested grid searches — expect tens of minutes to a few hours on a laptop, depending on core count.

---

## What the notebook does

**1 — Descriptors.** Reads `result/Mixed_exclude.xlsx`, parses each SMILES with `Chem.MolFromSmiles`, sanitizes explicitly, and computes every descriptor in `Descriptors.descList` (210 for RDKit 2024.3.5). Explicit hydrogens are deliberately *not* added. Written to `Case4_1.xlsx`.

**2 — Feature filtering.** Two stages: drop descriptors with a single unique value, then drop one member of every pair with `|Pearson r| > 0.80` (upper-triangle scan, keeping the earlier column). 210 → **27 descriptors**, written to `Case4_1_clean.xlsx`:

```
MaxAbsEStateIndex  MinAbsEStateIndex  MinEStateIndex  qed  SPS  MolWt
FpDensityMorgan1   BCUT2D_MWLOW  BCUT2D_LOGPHI  BCUT2D_MRLOW  BalabanJ
HallKierAlpha  Ipc  PEOE_VSA6  PEOE_VSA8  PEOE_VSA10  PEOE_VSA11
PEOE_VSA14  SMR_VSA4  SMR_VSA7  SlogP_VSA1  EState_VSA4  EState_VSA8
VSA_EState2  VSA_EState5  fr_Al_COO  fr_allylic_oxid
```

**3 — Target transformation.** `PowerTransformer(method='yeo-johnson')` on Cc, which brings skewness from +0.56 to +0.09 and kurtosis from −0.084 to −0.514. All reported metrics are computed after `pt.inverse_transform`, i.e. on the original Cc scale.

**4 — Feature scaling.** `RobustScaler(quantile_range=(10, 90))` followed by `MinMaxScaler()`, as a two-step `Pipeline`.

**5 — Model benchmarking.** Decision tree, random forest, gradient boosting and XGBoost, each under a grid search. The notebook sweeps the design space reported in the paper: StandardScaler vs. RobustScaler, raw vs. Yeo–Johnson target, 5- vs. 10-fold, non-nested vs. nested — one section per combination.

**6 — Final model.** Nested 10-fold CV (`KFold(n_splits=10, shuffle=True, random_state=42)` outer and inner). Inside each inner fold a wrapper selects the top-*n* features by SHAP importance, with *n* tuned alongside the XGBoost hyperparameters. The configuration is then refit and evaluated on an 80/20 split (`train_test_split(..., test_size=0.2, random_state=42)` → 50 train / 13 hold-out).

XGBoost grid actually searched in the notebook:

```python
{'n_estimators': [900, 1000, 1100, 1200],
 'max_depth': [3],
 'learning_rate': [0.01, 0.02],
 'subsample': [1.0, 0.8, 0.6],
 'reg_lambda': [1, 5, 10],
 'reg_alpha': [0, 0.1, 1],
 'gamma': [0, 0.1, 1],
 'min_child_weight': [5, 10]}
```

**7 — Interpretation.** `shap.TreeExplainer` on the final XGBoost model, giving mean(|SHAP|) rankings and beeswarm plots, plus partial-dependence curves for the highest-ranked descriptors.

## Headline results

| Metric | Training (n = 50) | Hold-out (n = 13) |
|---|---|---|
| R² | 1.000 | 0.85 |
| RMSE (Cc units) | 0.06 | 0.53 |
| MAE (Cc units) | 0.04 | 0.44 |

The hold-out MAE of 0.44 Cc units is comparable in magnitude to reported experimental uncertainty (≈ ±0.2 for alkyl ethoxylates; ≈ 0.2–1 for ionic surfactants), which is the basis for treating the model as a **pre-screening aid**, not a replacement for measurement. Because Cc enters the HLD equation additively with a coefficient of one, an error of ΔCc shifts the predicted HLD by the same amount — which matters most near HLD = 0.

The largest hold-out deviations are Tween-80 (AE = 0.99), C12EO5 (0.79) and C6EO3 (0.75).

---

## Reproducibility notes

Read these before reusing the pipeline or quoting the numbers.

**Preprocessing is fitted on all 63 molecules, before the train/hold-out split.** Constant-feature removal, the `|r| > 0.80` correlation prune, the Yeo–Johnson `PowerTransformer`, and the RobustScaler→MinMaxScaler pipeline are all `fit` on the complete dataset in cells 6–34 and 38 and 87. The nested CV and the 80/20 split both operate on the already-transformed matrix. The hold-out set is therefore not fully independent of preprocessing, and the 0.85 should be read as optimistic to an unquantified degree. A leakage-free version would move all four steps inside a `Pipeline` fitted per fold.

**The final configuration is chosen from the best-scoring outer fold.** `outer_results_df_XGB.loc[outer_results_df_XGB['Outer R2'].idxmax()]` picks the hyperparameters and feature subset from whichever outer fold scored highest. Those folds included the molecules that later became the hold-out set. Averaging across folds, or nesting the hold-out outside the CV entirely, would be the stricter choice.

**SHAP values are computed on training data.** `explainer_final.shap_values(X_train_full_optimal)` — 50 molecules, not the 13 hold-out ones. The saved plot is titled "Final Model (Training Data)" accordingly. Interpret the rankings as a description of what the fitted model learned, not as an out-of-sample attribution.

**Training R² = 1.000.** The final model interpolates its training set. With 50 samples and 1200 boosted trees this is expected, but it means the train/hold-out gap is the only usable evidence about generalization.

**Determinism.** `random_state=42` is set on every split, fold and estimator, so reruns on the same package versions reproduce exactly. Changing the RDKit or scikit-learn version can change the descriptor set or the fold assignment and therefore the numbers.

---

## Citation

Dulyadech, A.; Suriyapraphadilok, U.; Charoensaeng, A.; Sueviriyapan, N. *Interpretable QSPR Modeling of Surfactant Characteristic Curvature.* Manuscript in preparation.

Please cite the paper if you use this code or dataset. The Cc values themselves are compiled from 16 published sources — cite those originals when reusing the underlying measurements.

## Contact

Natthapong Sueviriyapan — natthapong.su@chula.ac.th
The Petroleum and Petrochemical College, Chulalongkorn University, Bangkok 10330, Thailand
