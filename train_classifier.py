from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from pipelines import (
    build_pipeline,
    get_preprocessing_configs,
    get_vectorizer_configs
)

from experiment_utils import (
    update_experiment_log,
    safe_text
)


def get_classifier_configs():

    return [

        (
            "MultinomialNB",
            MultinomialNB(alpha=1.0),
            "alpha=1.0"
        ),

        (
            "MultinomialNB",
            MultinomialNB(alpha=0.1),
            "alpha=0.1"
        ),

        (
            "LogisticRegression",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=0
            ),
            "C=1.0, max_iter=2000"
        ),

        (
            "LogisticRegression",
            LogisticRegression(
                C=5.0,
                max_iter=2000,
                random_state=0
            ),
            "C=5.0, max_iter=2000"
        ),

        (
            "LinearSVC",
            LinearSVC(
                C=1.0,
                random_state=0,
                dual="auto"
            ),
            "C=1.0"
        ),

        (
            "LinearSVC",
            LinearSVC(
                C=0.5,
                random_state=0,
                dual="auto"
            ),
            "C=0.5"
        ),
    ]


def aggregate_metrics(
    exp_id,
    model_name,
    split,
    y_true,
    y_pred
):

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "experiment_id": exp_id,
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def append_detailed_metrics(
    rows,
    confusion_rows,
    exp_id,
    model_name,
    split,
    y_true,
    y_pred,
    labels
):

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    # Metrics for each category.
    for label in labels:

        values = report[str(label)]

        rows.append({

            "experiment_id": exp_id,
            "model": model_name,
            "split": split,
            "class_label": label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": values["support"],

        })

    # Confusion matrix.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    for i, true_label in enumerate(labels):

        for j, predicted_label in enumerate(labels):

            confusion_rows.append({

                "experiment_id": exp_id,
                "model": model_name,
                "split": split,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": int(matrix[i, j]),

            })


def main():

    data_path = Path("labeled_documents.tsv")

    if not data_path.exists():

        raise FileNotFoundError(
            "Ejecuta primero: python split_data.py"
        )

    df = pd.read_csv(
        data_path,
        sep="\t"
    )

    required = {
        "document_id",
        "text",
        "category",
        "split"
    }

    if not required.issubset(df.columns):

        raise ValueError(
            f"labeled_documents.tsv debe contener {sorted(required)}"
        )

    # Separate partitions.
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()

    if min(
        len(train_df),
        len(dev_df),
        len(test_df)
    ) == 0:

        raise ValueError(
            "Train, dev y test deben contener documentos."
        )

    y_train = train_df["category"].astype(str).to_numpy()
    y_dev = dev_df["category"].astype(str).to_numpy()
    y_test = test_df["category"].astype(str).to_numpy()

    labels = sorted(
        df["category"].astype(str).unique()
    )

    prep_configs = get_preprocessing_configs()
    vector_configs = get_vectorizer_configs()
    classifier_configs = get_classifier_configs()

    experiments = []
    best_per_model = {}

    counter = 1

    # Run all supervised experiments.
    for norm_name, normalizer in prep_configs.items():

        for vec_name, vectorizer in vector_configs.items():

            for model_name, estimator, params in classifier_configs:

                exp_id = f"EXP_A_{counter:03d}"
                counter += 1

                pipe = build_pipeline(
                    normalizer,
                    vectorizer,
                    estimator
                )

                # Fit exclusively on train.
                pipe.fit(
                    train_df["text"].astype(str).tolist(),
                    y_train
                )

                # Evaluate on dev.
                dev_pred = pipe.predict(
                    dev_df["text"].astype(str).tolist()
                )

                metrics = aggregate_metrics(
                    exp_id,
                    model_name,
                    "dev",
                    y_dev,
                    dev_pred
                )

                experiments.append({

                    "experiment_id": exp_id,
                    "task": "classification",
                    "normalization": norm_name,
                    "representation": vec_name,
                    "method": model_name,
                    "hyperparameters": params,
                    "selection_metric": "macro_f1",
                    "dev_metric": metrics["macro_f1"],
                    "selected": False,
                    "selected_overall": False,
                    "notes": (
                        "Pipeline fitted on train; "
                        "evaluated on dev."
                    ),

                })

                current = {

                    "experiment_id": exp_id,
                    "model": model_name,
                    "normalization": norm_name,
                    "representation": vec_name,
                    "estimator": estimator,
                    "hyperparameters": params,
                    "dev_metrics": metrics,
                    "dev_predictions": dev_pred,

                }

                # Keep the best configuration for each algorithm.
                if (
                    model_name not in best_per_model
                    or metrics["macro_f1"]
                    > best_per_model[model_name]["dev_metrics"]["macro_f1"]
                ):

                    best_per_model[model_name] = current

    # Mark the best configuration of each classifier.
    for record in experiments:

        for model_name, best in best_per_model.items():

            if record["experiment_id"] == best["experiment_id"]:

                record["selected"] = True

    # Select the global winner.
    overall_best = max(
        best_per_model.values(),
        key=lambda item: item["dev_metrics"]["macro_f1"]
    )

    for record in experiments:

        if record["experiment_id"] == overall_best["experiment_id"]:

            record["selected_overall"] = True

    aggregate_rows = []
    per_class_rows = []
    confusion_rows = []
    prediction_rows = []

    # Save dev metrics for each classifier's best configuration.
    for model_name, best in best_per_model.items():

        aggregate_rows.append(
            best["dev_metrics"]
        )

        append_detailed_metrics(
            per_class_rows,
            confusion_rows,
            best["experiment_id"],
            model_name,
            "dev",
            y_dev,
            best["dev_predictions"],
            labels
        )

        for doc_id, gold, predicted in zip(
            dev_df["document_id"],
            y_dev,
            best["dev_predictions"]
        ):

            prediction_rows.append({

                "document_id": doc_id,
                "gold_label": gold,
                "predicted_label": predicted,
                "split": "dev",
                "model": model_name,
                "experiment_id": best["experiment_id"],

            })

    print(
        "BEST CONFIGURATION PER CLASSIFIER "
        "(DEV / MACRO-F1)"
    )

    for model_name, best in best_per_model.items():

        print(
            f"{model_name}: {best['experiment_id']} | "
            f"{best['normalization']} | "
            f"{best['representation']} | "
            f"macro-F1={best['dev_metrics']['macro_f1']:.4f}"
        )

    print("\nOVERALL WINNER:", overall_best["model"])
    print("Experiment:", overall_best["experiment_id"])

    # Refit the global winner using train + dev.
    train_dev_df = pd.concat(
        [train_df, dev_df],
        ignore_index=True
    )

    y_train_dev = (
        train_dev_df["category"]
        .astype(str)
        .to_numpy()
    )

    final_pipe = build_pipeline(

        get_preprocessing_configs()[
            overall_best["normalization"]
        ],

        get_vectorizer_configs()[
            overall_best["representation"]
        ],

        overall_best["estimator"]

    )

    final_pipe.fit(
        train_dev_df["text"].astype(str).tolist(),
        y_train_dev
    )

    # Final evaluation on test.
    test_pred = final_pipe.predict(
        test_df["text"].astype(str).tolist()
    )

    test_metrics = aggregate_metrics(
        "test",
        overall_best["model"],
        "test",
        y_test,
        test_pred
    )

    aggregate_rows.append(test_metrics)

    append_detailed_metrics(
        per_class_rows,
        confusion_rows,
        "test",
        overall_best["model"],
        "test",
        y_test,
        test_pred,
        labels
    )

    for doc_id, gold, predicted in zip(
        test_df["document_id"],
        y_test,
        test_pred
    ):

        prediction_rows.append({

            "document_id": doc_id,
            "gold_label": gold,
            "predicted_label": predicted,
            "split": "test",
            "model": overall_best["model"],
            "experiment_id": "test",

        })

    print("\nFINAL TEST RESULTS")

    print(
        classification_report(
            y_test,
            test_pred,
            labels=labels,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion matrix:")

    print(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                test_pred,
                labels=labels
            ),
            index=labels,
            columns=labels
        )
    )

    # Error analysis on dev for the global winner.
    best_dev_pred = overall_best["dev_predictions"]

    errors = []

    for (_, row), predicted in zip(
        dev_df.iterrows(),
        best_dev_pred
    ):

        if str(row["category"]) != str(predicted):

            errors.append({

                "document_id": row["document_id"],
                "gold_label": row["category"],
                "predicted_label": predicted,
                "text": safe_text(row["text"], 1000),
                "review_note": "",

            })

    pd.DataFrame(
        errors,
        columns=[
            "document_id",
            "gold_label",
            "predicted_label",
            "text",
            "review_note"
        ]
    ).to_csv(
        "classification_error_analysis.tsv",
        sep="\t",
        index=False
    )

    # Save all classification results.
    pd.DataFrame(aggregate_rows).to_csv(
        "classification_metrics.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame(per_class_rows).to_csv(
        "classification_per_class_metrics.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame(confusion_rows).to_csv(
        "classification_confusion_matrices.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame(prediction_rows).to_csv(
        "classification_predictions.tsv",
        sep="\t",
        index=False
    )

    # Update the experiment log without deleting clustering records.
    update_experiment_log(
        experiments,
        tasks_to_replace={"classification"}
    )

    # Save the final model.
    Path("results").mkdir(
        exist_ok=True
    )

    joblib.dump(
        final_pipe,
        "results/classifier_best.joblib"
    )

    metadata = {

        "experiment_id": overall_best["experiment_id"],
        "model": overall_best["model"],
        "normalization": overall_best["normalization"],
        "representation": overall_best["representation"],
        "hyperparameters": overall_best["hyperparameters"],
        "selection_metric": "dev_macro_f1",
        "test_metrics": test_metrics,

    }

    Path(
        "results/classifier_best_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nGenerated classification deliverables:")

    print("- classification_predictions.tsv")
    print("- classification_metrics.tsv")
    print("- classification_per_class_metrics.tsv")
    print("- classification_confusion_matrices.tsv")
    print("- classification_error_analysis.tsv")
    print("- experiments.tsv")
    print("- results/classifier_best.joblib")

    print(
        f"\nDev errors available for manual inspection: {len(errors)}"
    )


if __name__ == "__main__":
    main()
