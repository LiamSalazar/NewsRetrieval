
"""Create reproducible train/dev/test partitions for Stage II."""

from pathlib import Path
import json
import platform
import importlib.metadata

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0
INPUT_PATH = Path("data/news_raw_corpus.tsv")
OUTPUT_PATH = Path("labeled_documents.tsv")


def save_environment_info(output_dir="results"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    packages = {}

    for package in ("pandas", "numpy", "scikit-learn", "spacy", "joblib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "random_state": RANDOM_STATE,
        "packages": packages,
    }

    (output / "environment_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {INPUT_PATH}. "
            "Ejecuta este script desde la raíz del repositorio."
        )

    df = pd.read_csv(INPUT_PATH, sep="\t")

    required = {"id", "title", "category"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing)}"
        )

    body_col = (
        "description/body"
        if "description/body" in df.columns
        else "description"
    )

    if body_col not in df.columns:
        raise ValueError(
            "El corpus debe contener 'description/body' o 'description'."
        )

    initial_rows = len(df)

    df["category"] = df["category"].astype("string").str.strip()

    missing_label = (
        df["category"].isna()
        | df["category"].eq("")
    )

    dropped_missing_label = int(missing_label.sum())

    df = df.loc[~missing_label].copy()

    df["text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df[body_col].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    empty_text = df["text"].eq("")
    dropped_empty_text = int(empty_text.sum())

    df = df.loc[~empty_text].copy()

    if df["id"].isna().any():
        raise ValueError(
            "Hay documentos sin identificador 'id'."
        )

    if df["id"].duplicated().any():
        duplicates = (
            df.loc[df["id"].duplicated(), "id"]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Hay identificadores duplicados: {duplicates}"
        )

    class_counts = df["category"].value_counts()

    too_small = class_counts[class_counts < 3]

    if not too_small.empty:
        raise ValueError(
            "No es posible garantizar train/dev/test estratificados "
            "para estas clases (menos de 3 documentos): "
            f"{too_small.to_dict()}"
        )
    
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=temp_df["category"],
    )

    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"

    labeled = pd.concat(
        [train_df, dev_df, test_df],
        ignore_index=True
    )

    labeled = labeled[
        ["id", "text", "category", "split"]
    ].rename(
        columns={"id": "document_id"}
    )

    labeled.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    split_counts = (
        labeled["split"]
        .value_counts()
        .rename_axis("split")
        .reset_index(name="documents")
    )

    split_counts.to_csv(
        "split_summary.tsv",
        sep="\t",
        index=False
    )

    label_counts = (
        labeled.groupby(["category", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    label_counts.to_csv(
        "label_distribution.tsv",
        sep="\t",
        index=False
    )

    audit = pd.DataFrame([{
        "input_rows": initial_rows,
        "retained_labeled_nonempty_rows": len(labeled),
        "dropped_missing_category": dropped_missing_label,
        "dropped_empty_text": dropped_empty_text,
        "random_state": RANDOM_STATE,
        "train_ratio": len(train_df) / len(labeled),
        "dev_ratio": len(dev_df) / len(labeled),
        "test_ratio": len(test_df) / len(labeled),
    }])

    audit.to_csv(
        "split_audit.tsv",
        sep="\t",
        index=False
    )

    save_environment_info()


    print(split_counts.to_string(index=False))

    print("\nClass distribution by split:")

    print(
        pd.crosstab(
            labeled["category"],
            labeled["split"]
        )
    )

    print("\nGenerated:")
    print("- labeled_documents.tsv")
    print("- split_summary.tsv")
    print("- label_distribution.tsv")
    print("- split_audit.tsv")


if __name__ == "__main__":
    main()
