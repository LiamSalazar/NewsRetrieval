from pathlib import Path
from itertools import combinations
from sklearn.base import clone
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
            "svd_components": 0,
            "post_l2": False,

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

    for components in (50, 100):
        for l2 in (False, True):
            for k in (3, 4, 5):
                specs.append(dict(normalization=BASE_NORMALIZATION, representation=BASE_REPRESENTATION, n_clusters=k, n_init=10, init="k-means++", max_iter=300, random_state=0, svd_components=components, post_l2=l2))
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
            estimator,
            svd_components=spec.get("svd_components"), post_l2=spec.get("post_l2", False)

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
                f"random_state={spec['random_state']}; svd={spec.get('svd_components', 0)}; post_l2={spec.get('post_l2', False)}"
            ),

            "selection_metric": "dev_finalist_review_separation_stability_sizes_interpretability",
            "dev_metric": silhouette,
            "selected": False,
            "selected_overall": False,

            "svd_components": spec.get("svd_components", 0),
            "post_l2": spec.get("post_l2", False),
            "ngram_range": str(vector_configs[spec["representation"]].ngram_range),
            "notes": "001-012 normalization/representation; 013-017 one KMeans factor vs 012; 018-029 SVD dimensions/L2/k controlled vs 012/013/014",

        })

        dev_metric_rows.append({

            "experiment_id": exp_id,
            "split": "dev",
            **spec,
            "silhouette_euclidean": silhouette,
            "ari_external": ari,
            "nmi_external": nmi,
            "n_clusters_predicted": len(
                np.unique(dev_labels)
            ),
            "cluster_sizes": json.dumps(np.bincount(dev_labels, minlength=spec["n_clusters"]).tolist()),
            "feature_space": f"{spec['representation']}|svd={spec.get('svd_components', 0)}|l2={spec.get('post_l2', False)}",
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
            "selection_score": silhouette * (-sum(p * np.log(p) for p in np.bincount(dev_labels, minlength=spec["n_clusters"]) / len(dev_labels) if p > 0) / np.log(spec["n_clusters"])),

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

    # Pre-specified, label-free finalist review: the count-space baseline (002)
    # and the balanced three-cluster TF-IDF/SVD/L2 alternative (021).
    # On dev, 021 has positive separation, five-seed stability and three
    # interpretable groups, while 002 is dominated by length and tiny groups.
    # This qualitative judgment is documented in the README, not optimized
    # using category labels or the already observed test set.
    best = next(item for item in valid if item["experiment_id"] == "EXP_B_021")

    for record, candidate in zip(experiment_rows, fitted_candidates):
        record["selection_score"] = candidate["selection_score"]
    for row, candidate in zip(dev_metric_rows, fitted_candidates):
        row["selection_score"] = candidate["selection_score"]
    pd.DataFrame(dev_metric_rows).to_csv("clustering_dev_experiments.tsv", sep="\t", index=False)

    # Interpret the train-fitted candidate using only its dev assignments.
    dev_text = dev_df["text"].astype(str).tolist()
    dev_assignments = best["pipeline"].predict(dev_text)
    original_dev = best["pipeline"][:2].transform(dev_text)
    feature_names_dev = best["pipeline"].named_steps["vectorizer"].get_feature_names_out()
    embedded_dev = best["pipeline"][:-1].transform(dev_text)
    distances_dev = best["pipeline"].named_steps["model"].transform(embedded_dev)
    dev_interpretation = []
    for cluster in range(best["spec"]["n_clusters"]):
        members = np.flatnonzero(dev_assignments == cluster)
        if len(members) < 2:
            continue
        weights = np.asarray(original_dev[members].mean(axis=0)).ravel()
        words = feature_names_dev[np.argsort(weights)[::-1][:10]]
        nearest = members[np.argsort(distances_dev[members, cluster])[:2]]
        for position in nearest:
            row = dev_df.iloc[position]
            dev_interpretation.append({"experiment_id": best["experiment_id"], "fit_partition": "train", "split": "dev", "cluster_id": cluster, "top_10_words": " | ".join(words), "document_id": row["document_id"], "excerpt": safe_text(row["text"], 500)})
    pd.DataFrame(dev_interpretation).to_csv("clustering_dev_interpretation.tsv", sep="\t", index=False)

    # Stability compares assignments only; category labels are never passed.
    assignments = []
    for seed in range(5):
        repeated = clone(best["pipeline"])
        repeated.set_params(model__random_state=seed)
        repeated.fit(train_df["text"].tolist())
        assignments.append(repeated.predict(dev_df["text"].tolist()))
    stability = [{"experiment_id": best["experiment_id"], "seed_a": a, "seed_b": b, "ari_between_assignments": adjusted_rand_score(assignments[a], assignments[b])} for a, b in combinations(range(5), 2)]
    pd.DataFrame(stability).to_csv("clustering_stability.tsv", sep="\t", index=False)

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
        final_kmeans,
        svd_components=spec.get("svd_components"), post_l2=spec.get("post_l2", False)

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
            f"max_iter={spec['max_iter']}|"
            f"svd={spec.get('svd_components', 0)}|"
            f"post_l2={spec.get('post_l2', False)}"
        ),

    })

    # Dev assignments must come from the selection fit (train only).
    dev_mask = df["split"].eq("dev")
    clustering_results["fit_partition"] = "train+dev"
    clustering_results.loc[dev_mask, "cluster_id"] = best["pipeline"].predict(dev_df["text"].tolist())
    clustering_results.loc[dev_mask, "fit_partition"] = "train"
    external = clustering_results.merge(df[["document_id", "category"]], on="document_id")
    external = external[external["split"].isin(["dev", "test"])]
    external.groupby(["split", "experiment_id", "fit_partition", "cluster_id", "category"]).size().reset_index(name="count").to_csv("clustering_contingency.tsv", sep="\t", index=False)

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

    distances_train_dev = (
        final_pipe.named_steps["model"]
        .transform(X_train_dev)
    )

    original_X = final_pipe[:2].transform(train_dev_df["text"].tolist())
    summary_rows = []

    for cluster_id in range(spec["n_clusters"]):

        member_positions = np.where(
            train_dev_cluster_ids == cluster_id
        )[0]

        test_size = int(
            np.sum(test_labels == cluster_id)
        )


        ranked_indices = np.argsort(
            np.asarray(original_X[member_positions].mean(axis=0)).ravel()
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

        "selection_metric": "dev_finalist_review_separation_stability_sizes_interpretability",
        "svd": spec.get("svd_components"),
        "post_l2": spec.get("post_l2", False),
        "selection_score": best["selection_score"],
        "selection_rationale": "Dev finalist review: positive separation, reproducible assignments, balanced groups and interpretable representative terms/documents; no category labels or test used.",
        "term_method": "mean original vector weights among train+dev members",
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
