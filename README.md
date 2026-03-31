# EMOO4IMD: Ensemble Multi-Objective Optimization for Imbalanced Medical Data

This repository contains the evaluation pipeline for EMOO on imbalanced medical datasets, including:

1. Heart Failure Clinical Records
2. Pima Indians Diabetes
3. Mammographic Mass

The core EMOO method is maintained separately in the original repository. This repository focuses on dataset-specific preprocessing, experiment execution, performance evaluation, and result generation.

## What is included here
- dataset loading
- preprocessing
- train/test split
- calling the original EMOO optimizer
- Pareto front export
- test-set evaluation
- confusion matrix
- ROC/AUC
- CSV outputs for reproducibility

## What is not included here
- the original EMOO algorithm implementation
- DEAP / NSGA-II optimizer internals
- raw dataset files

## Repository structure
- `emoo_bridge.py` 
- `evaluation_utils.py` 
- `run_heart_failure.py`
- `run_pima.py`
- `run_mammographic.py`
- `data/README.md`

## Data requirements
Raw dataset files are not included in this repository. Before running the experiments, obtain the required datasets separately and place them in the following local paths:

- `./data/heart_failure_clinical_records_dataset.csv`
- `./data/diabetes.csv`
- `./data/mammographic_masses.csv`

## Generated outputs
The scripts are designed to generate result files during execution. Depending on the experiment, outputs may be written to directories such as:

- `./results/heart_failure/`
- `./results/pima/`
- `./results/mammographic/`

These outputs can include summary metrics, prediction files, confusion matrices, ROC plots, and Pareto-front exports.

## Important setup
Before running the experiments, edit `emoo_bridge.py` so it imports and calls the EMOO optimizer from the original repository.
