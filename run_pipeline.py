"""End-to-end Cc QSPR pipeline, as a script instead of a notebook:

    SMILES -> RDKit descriptors -> constant/correlation filtering ->
    scaling + Yeo-Johnson target transform -> nested-CV XGBoost ->
    final model -> SHAP / partial-dependence / parity plots.

This mirrors the notebook (`Interpretable QSPR Modeling of Surfactant
Characteristic Curvature.ipynb`) end to end, with each stage broken out
into a module under src/. Run from the repo root:

    python run_pipeline.py
"""
from src import config, data_loading, descriptors, feature_selection, interpretability, modeling


def main():
    print("1) RDKit descriptors...")
    if config.DESCRIPTORS_FILE.exists():
        full_df = data_loading.load_descriptor_dataset()
    else:
        full_df = descriptors.generate_descriptor_dataset()

    print("2) Constant/correlation feature filtering...")
    if config.CLEAN_FILE.exists():
        clean_df = data_loading.load_clean_dataset()
    else:
        clean_df = feature_selection.build_clean_dataset(full_df)
        clean_df.to_excel(config.CLEAN_FILE, index=False)

    X, y = data_loading.split_features_target(clean_df)

    print("3) Scaling features and transforming the target...")
    X_scaled = feature_selection.scale_features(X)
    y_transformed, power_transformer = feature_selection.transform_target(y)

    print("4) Nested cross-validated XGBoost (this takes a while)...")
    outer_results = modeling.nested_cv_xgboost(X_scaled, y_transformed)
    print(outer_results[["fold", "n_features", "r2", "rmse", "mae"]])
    print(f"Mean outer R2: {outer_results['r2'].mean():.3f}")

    print("5) Refitting the final model on a fresh train/hold-out split...")
    final = modeling.train_final_model(X_scaled, y_transformed, outer_results)
    print(f"Selected {len(final['feature_cols'])} features: {list(final['feature_cols'])}")

    # Map predictions back to the original Cc scale before reporting/plotting.
    y_train_pred = power_transformer.inverse_transform(
        final["model"].predict(final["X_train"]).reshape(-1, 1)
    ).ravel()
    y_holdout_pred = power_transformer.inverse_transform(
        final["model"].predict(final["X_holdout"]).reshape(-1, 1)
    ).ravel()
    y_train_orig = power_transformer.inverse_transform(final["y_train"].reshape(-1, 1)).ravel()
    y_holdout_orig = power_transformer.inverse_transform(final["y_holdout"].reshape(-1, 1)).ravel()

    metrics = interpretability.plot_parity(y_train_orig, y_train_pred, y_holdout_orig, y_holdout_pred)
    print(metrics)

    print("6) SHAP and partial dependence plots...")
    _, shap_values = interpretability.compute_shap_values(final["model"], final["X_train"])
    interpretability.plot_shap_summary(shap_values, final["X_train"])
    interpretability.plot_partial_dependence(final["model"], final["X_train"], shap_values)

    print(f"Done. Plots written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
