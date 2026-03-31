# evaluation_utils.py

import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_generation_log_csv(file_path, logbook):
    """
    Save DEAP logbook to CSV if available.
    Works if logbook entries behave like dicts.
    """
    rows = []
    for row in logbook:
        rows.append(dict(row))

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def save_hof_csv(file_path, hof):
    rows = []
    for i, ind in enumerate(hof, start=1):
        rows.append(
            {
                "solution_id": i,
                "n_estimators": ind[0],
                "max_depth": ind[1],
                "min_samples_split": ind[2],
                "criterion": ind[3],
                "fitness_accuracy": ind.fitness.values[0],
                "fitness_specificity": ind.fitness.values[1],
                "fitness_sensitivity": ind.fitness.values[2],
                "fitness_f1": ind.fitness.values[3],
                "fitness_std": ind.fitness.values[4],
            }
        )
    pd.DataFrame(rows).to_csv(file_path, index=False)


def build_base_estimators(hof, X_train, y_train, seed=42, rf_extra_kwargs=None):
    if rf_extra_kwargs is None:
        rf_extra_kwargs = {}

    base_estimators = []
    for i, ind in enumerate(hof):
        rf = RandomForestClassifier(
            n_estimators=int(ind[0]),
            max_depth=int(ind[1]),
            min_samples_split=int(ind[2]),
            criterion=str(ind[3]),
            random_state=seed,
            **rf_extra_kwargs,
        )
        rf.fit(X_train, y_train)
        base_estimators.append((f"rf_{i}", rf))

    return base_estimators


def build_stacking_ensemble(hof, X_train, y_train, seed=42, rf_extra_kwargs=None):
    base_estimators = build_base_estimators(
        hof, X_train, y_train, seed=seed, rf_extra_kwargs=rf_extra_kwargs
    )

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=AdaBoostClassifier(random_state=seed, algorithm="SAMME"),
        passthrough=True
    )
    stacker.fit(X_train, y_train)
    return stacker


def classwise_metrics(y_true, y_pred):
    labels = sorted(np.unique(y_true))
    acc_scores = []
    sens_scores = []
    spec_scores = []
    f1_scores = []
    weights = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        tn = np.sum((y_pred != c) & (y_true != c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

        acc_scores.append(accuracy)
        sens_scores.append(sensitivity)
        spec_scores.append(specificity)
        f1_scores.append(f1)
        weights.append(np.sum(y_true == c))

    weights = np.asarray(weights, dtype=float)

    final_accuracy = np.average(acc_scores, weights=weights)
    final_sensitivity = np.average(sens_scores, weights=weights)
    final_specificity = np.average(spec_scores, weights=weights)
    final_f1 = np.average(f1_scores, weights=weights)
    final_std = np.std([final_accuracy, final_specificity, final_sensitivity, final_f1])

    return {
        "accuracy": final_accuracy,
        "specificity": final_specificity,
        "sensitivity": final_sensitivity,
        "f1_score": final_f1,
        "std": final_std,
        "labels": labels,
        "weights": weights,
    }


def save_predictions(file_path, y_true, y_pred, y_proba=None, classes=None):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    if y_proba is not None and classes is not None:
        for idx, c in enumerate(classes):
            df[f"p_class_{c}"] = y_proba[:, idx]

    df.to_csv(file_path, index=False)


def compute_weighted_one_vs_rest_auc(y_true, y_proba, classes):
    """
    Weighted one-vs-rest AUC for binary or multiclass.
    """
    auc_scores = []
    weights = []

    y_true = np.asarray(y_true)

    for idx, c in enumerate(classes):
        y_true_bin = (y_true == c).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, idx])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        weights.append(np.sum(y_true == c))

    if not auc_scores:
        return np.nan

    return float(np.average(auc_scores, weights=np.asarray(weights, dtype=float)))


def plot_binary_roc(file_path, y_true, y_proba, positive_label=1, title="ROC Curve"):
    y_true_bin = (np.asarray(y_true) == positive_label).astype(int)

    if len(np.unique(y_true_bin)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true_bin, y_proba)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    return float(auc_score)


def plot_multiclass_ovr_rocs(output_dir, y_true, y_proba, classes, title_prefix="ROC"):
    auc_scores = []

    for idx, c in enumerate(classes):
        y_true_bin = (np.asarray(y_true) == c).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, idx])
        auc_score = auc(fpr, tpr)
        auc_scores.append((c, auc_score))

        plt.figure()
        plt.plot(fpr, tpr, label=f"Class {c} AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} - Class {c}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"roc_class_{c}.png")
        plt.close()

    return auc_scores


def save_confusion_matrix_csv(file_path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(file_path, index=False)


def save_summary_csv(file_path, dataset_name, results, auc_value):
    row = {
        "dataset": dataset_name,
        "accuracy": results["accuracy"],
        "specificity": results["specificity"],
        "sensitivity": results["sensitivity"],
        "f1_score": results["f1_score"],
        "std": results["std"],
        "weighted_auc_ovr": auc_value,
    }
    pd.DataFrame([row]).to_csv(file_path, index=False)


def evaluate_hof_ensemble(
    hof,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir,
    dataset_name,
    seed=42,
    rf_extra_kwargs=None,
):
    ensure_dir(output_dir)

    stacker = build_stacking_ensemble(
        hof=hof,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
        rf_extra_kwargs=rf_extra_kwargs,
    )

    y_pred = stacker.predict(X_test)
    y_proba = stacker.predict_proba(X_test)
    classes = list(stacker.classes_)

    results = classwise_metrics(y_test, y_pred)
    auc_value = compute_weighted_one_vs_rest_auc(y_test, y_proba, classes)

    save_predictions(
        Path(output_dir) / "test_predictions.csv",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
    )

    save_confusion_matrix_csv(
        Path(output_dir) / "confusion_matrix.csv",
        y_true=y_test,
        y_pred=y_pred,
    )

    save_summary_csv(
        Path(output_dir) / "summary_metrics.csv",
        dataset_name=dataset_name,
        results=results,
        auc_value=auc_value,
    )

    if len(classes) == 2:
        # probability column corresponding to positive label = last class
        positive_label = classes[-1]
        positive_idx = classes.index(positive_label)
        plot_binary_roc(
            Path(output_dir) / "roc_curve.png",
            y_true=y_test,
            y_proba=y_proba[:, positive_idx],
            positive_label=positive_label,
            title=f"{dataset_name} ROC Curve",
        )
    else:
        plot_multiclass_ovr_rocs(
            output_dir=output_dir,
            y_true=y_test,
            y_proba=y_proba,
            classes=classes,
            title_prefix=dataset_name,
        )

    return {
        "stacker": stacker,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "results": results,
        "auc": auc_value,
        "classes": classes,
    }