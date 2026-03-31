# run_pima.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from emoo_bridge import run_emoo_optimizer
from evaluation_utils import (
    ensure_dir,
    save_generation_log_csv,
    save_hof_csv,
    evaluate_hof_ensemble,
)


SEED = 42


def load_pima(data_path):
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X = data.drop(columns=["Outcome"])
    y = data["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def main():
    dataset_name = "Pima Indians Diabetes"
    data_path = "./data/diabetes.csv"
    output_dir = Path("./results/pima")

    ensure_dir(output_dir)

    X_train, X_test, y_train, y_test = load_pima(data_path)

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
        rf_extra_kwargs={},
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
        rf_extra_kwargs={},
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