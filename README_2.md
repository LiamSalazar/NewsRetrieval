
# NewsRetrieval — Stage II

## Overview

This implementation includes:

- Supervised news classification.
- Unsupervised K-Means clustering.

K-Means is the only unsupervised method used in this stage.

## Requirements

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Execution order

Run all commands from the repository root.

### 1. Data splitting

```bash
python split_data.py
```

Generates:

- labeled_documents.tsv
- split_summary.tsv
- label_distribution.tsv
- split_audit.tsv

### 2. Supervised classification

```bash
python train_classifier.py
```

The experiments include:

- Multinomial Naive Bayes.
- Logistic Regression.
- Linear SVC.
- Three normalization configurations.
- Four vector representations.
- Hyperparameter comparisons.

Model selection uses macro-F1 on the development partition.

The selected model is refitted using train + dev and evaluated on test.

Generated files:

- classification_predictions.tsv
- classification_metrics.tsv
- classification_per_class_metrics.tsv
- classification_confusion_matrices.tsv
- classification_error_analysis.tsv

### 3. Unsupervised learning

```bash
python train_unsupervised.py
```

Only K-Means is implemented.

Experiments compare:

- Text normalization.
- Vector representations.
- Number of clusters.
- Initialization method.
- Number of initializations.
- Maximum iterations.

Model selection uses silhouette score on dev.

ARI and NMI are used only as external evaluation metrics.

Generated files:

- clustering_results.tsv
- clustering_metrics.tsv
- cluster_summary.tsv

## Experiment tracking

Both training scripts update:

- experiments.tsv

Existing records from the other task are preserved.

## Model artifacts

The following models and metadata are saved in results/:

- classifier_best.joblib
- classifier_best_metadata.json
- kmeans_best.joblib
- kmeans_best_metadata.json
- environment_info.json

## Methodological considerations

- Train is used for initial model fitting.
- Dev is used for model selection.
- Test is reserved for final evaluation.
- Category labels are never used to train K-Means.
- SVD is not applied, preserving the original feature space for cluster interpretation.

## Final report

The team must still interpret the experimental results, review classification errors, inspect representative cluster documents, discuss limitations, and prepare the final report.
