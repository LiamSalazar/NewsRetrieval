import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from pipelines import get_preprocessing_configs, get_vectorizer_configs


def get_classifier_configs():
    """Returns the 3 traditional classifiers with hyperparameter variations."""
    return [
        ("MultinomialNB", MultinomialNB(alpha=1.0), "alpha=1.0"),
        ("MultinomialNB", MultinomialNB(alpha=0.1), "alpha=0.1"),
        ("LogisticRegression", LogisticRegression(C=1.0, max_iter=1000, random_state=0), "C=1.0, max_iter=1000"),
        ("LogisticRegression", LogisticRegression(C=5.0, max_iter=1000, random_state=0), "C=5.0, max_iter=1000"),
        ("LinearSVC", LinearSVC(C=1.0, random_state=0, dual="auto"), "C=1.0"),
        ("LinearSVC", LinearSVC(C=0.5, random_state=0, dual="auto"), "C=0.5"),
    ]


def compute_metrics_row(exp_id, model_name, split_name, y_true, y_pred):
    """Computes macro metrics and returns a dictionary formatted for TSV export."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "experiment_id": exp_id,
        "model": model_name,
        "split": split_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4)
    }


def main():
    df = pd.read_csv("labeled_documents.tsv", sep="\t")

    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()

    y_train = train_df["category"].values
    y_dev = dev_df["category"].values
    y_test = test_df["category"].values

    prep_configs = get_preprocessing_configs()
    vec_configs = get_vectorizer_configs()
    clf_configs = get_classifier_configs()

    normalized_data = {}
    for norm_name, normalizer in prep_configs.items():
        print(f"   Processing {norm_name}...")
        normalized_data[norm_name] = {
            "train": normalizer.transform(train_df["text"].tolist()),
            "dev": normalizer.transform(dev_df["text"].tolist()),
            "test": normalizer.transform(test_df["text"].tolist()),
        }

    experiments_log = []
    best_per_model = {}
    overall_best = {"macro_f1": -1.0}

    exp_counter = 1
    for norm_name in prep_configs.keys():
        X_train_text = normalized_data[norm_name]["train"]
        X_dev_text = normalized_data[norm_name]["dev"]

        for vec_name, vectorizer in vec_configs.items():
            for model_name, clf, params_str in clf_configs:
                exp_id = f"EXP_A_{exp_counter:03d}"
                exp_counter += 1

                pipe = Pipeline([
                    ("vec", vectorizer),
                    ("clf", clf)
                ])
                pipe.fit(X_train_text, y_train)
                y_dev_pred = pipe.predict(X_dev_text)

                _, _, macro_f1, _ = precision_recall_fscore_support(
                    y_dev, y_dev_pred, average="macro", zero_division=0
                )
                macro_f1 = round(float(macro_f1), 4)

                exp_record = {
                    "experiment_id": exp_id,
                    "task": "classification",
                    "normalization": norm_name,
                    "representation": vec_name,
                    "method": model_name,
                    "hyperparameters": params_str,
                    "dev_metric": macro_f1,
                    "selected": False
                }
                experiments_log.append(exp_record)

                if model_name not in best_per_model or macro_f1 > best_per_model[model_name]["macro_f1"]:
                    best_per_model[model_name] = {
                        "exp_id": exp_id,
                        "norm_name": norm_name,
                        "vec_name": vec_name,
                        "model_name": model_name,
                        "clf": clf,
                        "params_str": params_str,
                        "macro_f1": macro_f1,
                        "y_dev_pred": y_dev_pred
                    }

                if macro_f1 > overall_best["macro_f1"]:
                    overall_best = best_per_model[model_name].copy()

    for rec in experiments_log:
        if rec["experiment_id"] == overall_best["exp_id"]:
            rec["selected"] = True

    print("BEST RESULTS PER ALGORITHM ON DEV (Selection Metric: Macro-F1)")
    metrics_rows = []
    predictions_rows = []

    for model_name, info in best_per_model.items():
        print(f"- {model_name:20s}, ID: {info['exp_id']}, {info['norm_name']} + {info['vec_name']}, Macro-F1: {info['macro_f1']:.4f}")
        metrics_rows.append(
            compute_metrics_row(info["exp_id"], model_name, "dev", y_dev, info["y_dev_pred"])
        )
        for doc_id, gold, pred in zip(dev_df["document_id"], y_dev, info["y_dev_pred"]):
            predictions_rows.append({
                "experiment_id": info["exp_id"],
                "model": model_name,
                "document_id": doc_id,
                "gold_label": gold,
                "predicted_label": pred,
                "split": "dev"
            })

    print(f"SELECTED WINNING MODEL: {overall_best['model_name']} ({overall_best['exp_id']})")
    print(f"Configuration: {overall_best['norm_name']} + {overall_best['vec_name']} ({overall_best['params_str']})")

    X_train_dev_text = (
        normalized_data[overall_best["norm_name"]]["train"]
        + normalized_data[overall_best["norm_name"]]["dev"]
    )
    y_train_dev = np.concatenate([y_train, y_dev])
    X_test_text = normalized_data[overall_best["norm_name"]]["test"]

    final_pipe = Pipeline([
        ("vec", vec_configs[overall_best["vec_name"]]),
        ("clf", overall_best["clf"])
    ])
    final_pipe.fit(X_train_dev_text, y_train_dev)
    y_test_pred = final_pipe.predict(X_test_text)

    test_metrics = compute_metrics_row(
        overall_best["exp_id"], overall_best["model_name"], "test", y_test, y_test_pred
    )
    metrics_rows.append(test_metrics)

    for doc_id, gold, pred in zip(test_df["document_id"], y_test, y_test_pred):
        predictions_rows.append({
            "experiment_id": overall_best["exp_id"],
            "model": overall_best["model_name"],
            "document_id": doc_id,
            "gold_label": gold,
            "predicted_label": pred,
            "split": "test"
        })

    print("\nFINAL PERFORMANCE ON TEST:")
    print(f"Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"Macro-Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Macro-F1:        {test_metrics['macro_f1']:.4f}\n")

    print("Per-class Classification Report on TEST:")
    print(classification_report(y_test, y_test_pred, digits=4))

    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    print("Confusion Matrix on TEST:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    pd.DataFrame(experiments_log).to_csv("experiments.tsv", sep="\t", index=False)
    pd.DataFrame(predictions_rows).to_csv("classification_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(metrics_rows).to_csv("classification_metrics.tsv", sep="\t", index=False)

    print("\nGenerated TSV deliverables:")
    print("- experiments.tsv")
    print("- classification_predictions.tsv")
    print("- classification_metrics.tsv")
    print("SAMPLE MISCLASSIFIED DOCUMENTS ON DEV (Error Analysis)")
    dev_errors = dev_df[y_dev != overall_best["y_dev_pred"]].copy()
    dev_errors["predicted"] = overall_best["y_dev_pred"][y_dev != overall_best["y_dev_pred"]]
    for _, row in dev_errors.head(10).iterrows():
        print(f"ID: {row['document_id']}, Gold: {row['category']}, Predicted: {row['predicted']}")
        print(f"Text: {row['text'][:140]}...\n")


if __name__ == "__main__":
    main()