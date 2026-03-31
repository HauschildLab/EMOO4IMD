# run_mammographic.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from emoo_bridge import run_emoo_optimizer
from evaluation_utils import (
    ensure_dir,
    save_generation_log_csv,
    save_hof_csv,
    evaluate_hof_ensemble,
)


SEED = 42


def load_mammographic(data_path):
    data = pd.read_csv(data_path)

    data.columns = data.columns.str.strip().str.replace(" ", "_").str.lower()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # IMPORTANT:
    # In your original code these were categorical
    categorical_cols = ["bi-rads", "shape", "margin", "density"]
    numerical_cols = ["age"]

    # drop rows with missing target
    data = data.dropna(subset=["severity"]).reset_index(drop=True)

    # cast categorical integer-coded columns to string before one-hot encoding
    for col in categorical_cols:
        data[col] = data[col].astype("Int64").astype("string")

    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X = data.drop(columns=["severity"])
    y = data["severity"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    output_columns = numerical_cols + cat_feature_names.tolist()

    X_train_df = pd.DataFrame(
        X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed,
        columns=output_columns,
        index=X_train.index
    )

    X_test_df = pd.DataFrame(
        X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed,
        columns=output_columns,
        index=X_test.index
    )

    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def main():
    dataset_name = "Mammographic Mass"
    data_path = "./data/mammographic_masses.csv"
    output_dir = Path("./results/mammographic")

    ensure_dir(output_dir)

    X_train, X_test, y_train, y_test = load_mammographic(data_path)

    hof, logbook = run_emoo_optimizer(
        X_train=X_train,
        y_train=y_train,
        seed=SEED,
        n_pop=50,
        ngen=100,
        n_estimators_range=np.arange(100, 300, 10),
        max_depth_choices=np.arange(5, 31),
        min_samples_split_range=np.arange(2, 21),
        criterion_choices=["gini", "entropy", "log_loss"],
        rf_extra_kwargs={"class_weight": "balanced"},
    )

    save_hof_csv(output_dir / "pareto_front.csv", hof)
    save_generation_log_csv(output_dir / "generation_log.csv", logbook)

    evaluation = evaluate_hof_ensemble(
        hof=hof,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        dataset_name=dataset_name,
        seed=SEED,
        rf_extra_kwargs={"class_weight": "balanced"},
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Accuracy    : {evaluation['results']['accuracy']:.4f}")
    print(f"Specificity : {evaluation['results']['specificity']:.4f}")
    print(f"Sensitivity : {evaluation['results']['sensitivity']:.4f}")
    print(f"F1 Score    : {evaluation['results']['f1_score']:.4f}")
    print(f"Std         : {evaluation['results']['std']:.4f}")
    print(f"Weighted AUC: {evaluation['auc']:.4f}")


if __name__ == "__main__":
    main()