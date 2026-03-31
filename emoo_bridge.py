# emoo_bridge.py
"""
Bridge to the ORIGINAL EMOO repository.

This file is the only place in the evaluation repository that knows
how to call the EMOO optimizer from the original codebase.

IMPORTANT:
1) Clone/download the original EMOO repo locally.
2) Make sure Python can import it.
3) Replace the import/function below with your real one.

Expected output from run_emoo_optimizer():
    hof, logbook

Where:
- hof is an iterable of Pareto-optimal individuals
- each individual is like:
    [n_estimators, max_depth, min_samples_split, criterion]
"""

# -------------------------------------------------------------------
# Replace the import below with your actual function from the old repo.
#
# Example:
# from emoo_core import run_emoo_rf
#
# That function should optimize on X_train, y_train and return hof, logbook
# -------------------------------------------------------------------

def run_emoo_optimizer(
    X_train,
    y_train,
    seed=42,
    n_pop=50,
    ngen=100,
    n_estimators_range=None,
    max_depth_choices=None,
    min_samples_split_range=None,
    criterion_choices=None,
    rf_extra_kwargs=None,
):
    """
    Thin wrapper that should call the optimizer in the ORIGINAL EMOO repo.

    You must replace the body of this function with the real call to your
    original EMOO implementation.
    """

    raise NotImplementedError(
        "\nEdit emoo_bridge.py and connect this function to the original EMOO repository.\n"
        "The evaluation repo should NOT re-implement EMOO.\n"
        "It should only call the existing optimizer and evaluate the returned Pareto front.\n"
    )