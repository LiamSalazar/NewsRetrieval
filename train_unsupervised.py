from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

from pipelines import (
    build_pipeline,
    get_preprocessing_configs,
    get_vectorizer_configs
)

from experiment_utils import (
    update_experiment_log,
    safe_text
)


RANDOM_STATE = 0

BASE_NORMALIZATION = "norm_3_lemma"
BASE_REPRESENTATION = "vec_tfidf_1_2"

SILHOUETTE_SAMPLE_SIZE = 1000


def build_experiment_specs():

    """
    Generate controlled K-Means experiments.

    First: compare preprocessing and representations.
    Then: compare K-Means hyperparameters.
    """

    specs = []
    seen = set()

    def add(
        norm,
        vec,
        k=4,
        n_init=10,
        init="k-means++",
        max_iter=300
    ):

        key = (
            norm,
            vec,
            k,
            n_init,
            init,
            max_iter,
            RANDOM_STATE
        )

        if key in seen:
            return

        seen.add(key)

        specs.append({

            "normalization": norm,
            "representation": vec,
            "n_clusters": k,
            "n_init": n_init,
            "init": init,
            "max_iter": max_iter,
            "random_state": RANDOM_STATE,

        })

    for norm in get_preprocessing_configs():

        for vec in get_vectorizer_configs():

            add(norm, vec)


    add(
        BASE_NORMALIZATION,
        BASE_REPRESENTATION,
        k=3
    )

    add(
        BASE_NORMALIZATION,
        BASE_REPRESENTATION,
        k=5
    )

    add(
        BASE_NORMALIZATION,
        BASE_REPRESENTATION,
        n_init=20
    )

    add(
        BASE_NORMALIZATION,
        BASE_REPRESENTATION,
        init="random"
    )

    add(
        BASE_NORMALIZATION,
        BASE_REPRESENTATION,
        max_iter=500
    )

    return specs


def calculate_silhouette(X, labels):

    """
    Calculate silhouette using Euclidean distance.

    Return NaN when silhouette cannot be calculated.
    """

    unique = np.unique(labels)

    if len(unique) < 2 or len(unique) >= X.shape[0]:

        return np.nan

    sample_size = min(
        SILHOUETTE_SAMPLE_SIZE,
        X.shape[0]
    )

    try:

        return float(
            silhouette_score(
                X,
                labels,
                metric="euclidean",
                sample_size=(
                    sample_size
                    if sample_size < X.shape[0]
                    else None
                ),
                random_state=RANDOM_STATE
            )
        )

    except ValueError:

        return np.nan


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

    prep_configs = get_preprocessing_configs()
    vector_configs = get_vectorizer_configs()

    specs = build_experiment_specs()

    experiment_rows = []
    dev_metric_rows = []
    fitted_candidates = []

    for index, spec in enumerate(specs, start=1):

        exp_id = f"EXP_B_{index:03d}"

        estimator = KMeans(

            n_clusters=spec["n_clusters"],
            init=spec["init"],
            n_init=spec["n_init"],
            max_iter=spec["max_iter"],
            random_state=spec["random_state"]

        )

        pipe = build_pipeline(

            prep_configs[spec["normalization"]],
            vector_configs[spec["representation"]],
            estimator

        )

        pipe.fit(
            train_df["text"].astype(str).tolist()
        )

        dev_labels = pipe.predict(
            dev_df["text"].astype(str).tolist()
        )

        X_dev = pipe[:-1].transform(
            dev_df["text"].astype(str).tolist()
        )

        silhouette = calculate_silhouette(
            X_dev,
            dev_labels
        )

        y_dev = (
            dev_df["category"]
            .astype(str)
            .to_numpy()
        )

        ari = adjusted_rand_score(
            y_dev,
            dev_labels
        )

        nmi = normalized_mutual_info_score(
            y_dev,
            dev_labels
        )

        experiment_rows.append({

            "experiment_id": exp_id,
            "task": "clustering",
            "normalization": spec["normalization"],
            "representation": spec["representation"],
            "method": "KMeans",

            "hyperparameters": (
                f"n_clusters={spec['n_clusters']}; "
                f"n_init={spec['n_init']}; "
                f"init={spec['init']}; "
                f"max_iter={spec['max_iter']}; "
                f"random_state={spec['random_state']}"
            ),

            "selection_metric": "dev_silhouette_euclidean",
            "dev_metric": silhouette,
            "selected": False,
            "selected_overall": False,

            "notes": (
                "No SVD; original word-feature space "
                "retained for interpretation."
            ),

        })

        dev_metric_rows.append({

            "experiment_id": exp_id,
            "split": "dev",
            "silhouette_euclidean": silhouette,
            "ari_external": ari,
            "nmi_external": nmi,
            "n_clusters_predicted": len(
                np.unique(dev_labels)
            ),
            "silhouette_sample_size": min(
                SILHOUETTE_SAMPLE_SIZE,
                len(dev_df)
            ),

        })

        fitted_candidates.append({

            "experiment_id": exp_id,
            "spec": spec,
            "pipeline": pipe,
            "silhouette": silhouette,

        })

        print(
            f"{exp_id}: "
            f"{spec['normalization']} | "
            f"{spec['representation']} | "
            f"k={spec['n_clusters']} | "
            f"silhouette={silhouette:.4f}"
        )

    valid = [

        item
        for item in fitted_candidates
        if np.isfinite(item["silhouette"])

    ]

    if not valid:

        raise RuntimeError(
            "No se pudo calcular silhouette en dev para "
            "ninguna configuración. Revisa el tamaño de dev "
            "y la cantidad de clusters predichos."
        )

    best = max(
        valid,
        key=lambda item: item["silhouette"]
    )

    for record in experiment_rows:

        if record["experiment_id"] == best["experiment_id"]:

            record["selected"] = True
            record["selected_overall"] = True

    train_dev_df = pd.concat(
        [train_df, dev_df],
        ignore_index=True
    )

    spec = best["spec"]

    final_kmeans = KMeans(

        n_clusters=spec["n_clusters"],
        init=spec["init"],
        n_init=spec["n_init"],
        max_iter=spec["max_iter"],
        random_state=spec["random_state"]

    )

    final_pipe = build_pipeline(

        prep_configs[spec["normalization"]],
        vector_configs[spec["representation"]],
        final_kmeans

    )

    final_pipe.fit(
        train_dev_df["text"].astype(str).tolist()
    )

    all_labels = final_pipe.predict(
        df["text"].astype(str).tolist()
    )

    clustering_results = pd.DataFrame({

        "document_id": df["document_id"],
        "split": df["split"],
        "cluster_id": all_labels,
        "method": "KMeans",
        "experiment_id": best["experiment_id"],

        "configuration": (
            f"{spec['normalization']}|"
            f"{spec['representation']}|"
            f"k={spec['n_clusters']}|"
            f"n_init={spec['n_init']}|"
            f"init={spec['init']}|"
            f"max_iter={spec['max_iter']}"
        ),

    })

    clustering_results.to_csv(
        "clustering_results.tsv",
        sep="\t",
        index=False
    )

    test_texts = (
        test_df["text"]
        .astype(str)
        .tolist()
    )

    test_labels = final_pipe.predict(
        test_texts
    )

    X_test = final_pipe[:-1].transform(
        test_texts
    )

    test_silhouette = calculate_silhouette(
        X_test,
        test_labels
    )

    y_test = (
        test_df["category"]
        .astype(str)
        .to_numpy()
    )

    test_ari = adjusted_rand_score(
        y_test,
        test_labels
    )

    test_nmi = normalized_mutual_info_score(
        y_test,
        test_labels
    )

    dev_best_row = next(

        row
        for row in dev_metric_rows
        if row["experiment_id"] == best["experiment_id"]

    )

    final_metric_rows = [

        dev_best_row,

        {

            "experiment_id": "test",
            "split": "test",
            "silhouette_euclidean": test_silhouette,
            "ari_external": test_ari,
            "nmi_external": test_nmi,
            "n_clusters_predicted": len(
                np.unique(test_labels)
            ),
            "silhouette_sample_size": min(
                SILHOUETTE_SAMPLE_SIZE,
                len(test_df)
            ),

        }

    ]

    pd.DataFrame(final_metric_rows).to_csv(
        "clustering_metrics.tsv",
        sep="\t",
        index=False
    )

    
    X_train_dev = final_pipe[:-1].transform(
        train_dev_df["text"].astype(str).tolist()
    )

    train_dev_cluster_ids = (
        final_pipe.named_steps["model"]
        .predict(X_train_dev)
    )

    centers = (
        final_pipe.named_steps["model"]
        .cluster_centers_
    )

    feature_names = (
        final_pipe.named_steps["vectorizer"]
        .get_feature_names_out()
    )

    # Distancias calculadas exclusivamente con train + dev.
    # No utilizamos documentos del conjunto test para
    # seleccionar ejemplos representativos.

    distances_train_dev = (
        final_pipe.named_steps["model"]
        .transform(X_train_dev)
    )

    summary_rows = []

    for cluster_id in range(spec["n_clusters"]):

        member_positions = np.where(
            train_dev_cluster_ids == cluster_id
        )[0]

        test_size = int(
            np.sum(test_labels == cluster_id)
        )

        # Obtener los términos con mayor peso en el centroide.
        # Se excluyen términos formados únicamente por stopwords.

        ranked_indices = np.argsort(
            centers[cluster_id]
        )[::-1]

        top_words = []

        for index in ranked_indices:

            term = feature_names[index]

            tokens = term.lower().split()

            if all(
                token in ENGLISH_STOP_WORDS
                for token in tokens
            ):
                continue

            top_words.append(term)

            if len(top_words) == 10:
                break

        # Seleccionar los documentos más cercanos al centroide.
        # Únicamente se consideran documentos de train + dev.

        if len(member_positions):

            nearest_order = member_positions[
                np.argsort(
                    distances_train_dev[
                        member_positions,
                        cluster_id
                    ]
                )
            ][:2]

            representative_ids = (
                train_dev_df.iloc[nearest_order]["document_id"]
                .astype(str)
                .tolist()
            )

            representative_texts = [

                safe_text(value, 500)

                for value in (
                    train_dev_df.iloc[nearest_order]["text"].tolist()
                )

            ]

        else:

            representative_ids = []
            representative_texts = []

        summary_rows.append({

            "cluster_id": cluster_id,

            "size_train_dev": int(
                len(member_positions)
            ),

            "size_test": test_size,

            "top_10_words": " | ".join(top_words),

            "representative_document_ids": (
                " | ".join(representative_ids)
            ),

            "representative_documents": (
                " || ".join(representative_texts)
            ),

        })

    pd.DataFrame(summary_rows).to_csv(
        "cluster_summary.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame(summary_rows).to_csv(
        "cluster_summary.tsv",
        sep="\t",
        index=False
    )

    update_experiment_log(
        experiment_rows,
        tasks_to_replace={
            "clustering",
            "topic_modeling"
        }
    )

    Path("results").mkdir(
        exist_ok=True
    )

    joblib.dump(
        final_pipe,
        "results/kmeans_best.joblib"
    )

    metadata = {

        "experiment_id": best["experiment_id"],
        "method": "KMeans",

        "normalization": spec["normalization"],
        "representation": spec["representation"],

        "hyperparameters": {

            "n_clusters": spec["n_clusters"],
            "n_init": spec["n_init"],
            "init": spec["init"],
            "max_iter": spec["max_iter"],
            "random_state": spec["random_state"],

        },

        "selection_metric": "dev_silhouette_euclidean",
        "svd": None,
        "test_metrics": final_metric_rows[-1],

    }

    Path(
        "results/kmeans_best_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nSELECTED K-MEANS CONFIGURATION")

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        )
    )

    print("\nGenerated:")

    print("- clustering_results.tsv")
    print("- clustering_metrics.tsv")
    print("- cluster_summary.tsv")
    print("- experiments.tsv")
    print("- results/kmeans_best.joblib")


if __name__ == "__main__":
    main()
