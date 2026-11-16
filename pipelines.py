
"""Reusable text preprocessing and sklearn Pipeline factories."""

import re
from pathlib import Path

import spacy

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.pipeline import Pipeline
from joblib import Memory

_NLP = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]
)


class TextNormalizer(BaseEstimator, TransformerMixin):

    """
    baseline:
        Lowercase and whitespace normalization.

    clean_stopwords:
        Remove punctuation, numbers and stopwords.

    full_lemma:
        Remove punctuation, numbers and stopwords,
        then apply lemmatization.
    """

    def __init__(self, mode="baseline", batch_size=128):
        self.mode = mode
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        texts = [
            "" if value is None else str(value)
            for value in X
        ]

        if self.mode == "baseline":

            return [
                re.sub(r"\s+", " ", text).strip().lower()
                for text in texts
            ]

        cleaned = [
            re.sub(
                r"[^A-Za-z\s']",
                " ",
                text
            ).replace("'", " ")
            for text in texts
        ]

        output = []

        for doc in _NLP.pipe(
            cleaned,
            batch_size=self.batch_size
        ):

            tokens = []

            for token in doc:

                if (
                    token.is_space
                    or token.is_punct
                    or token.like_num
                    or token.is_stop
                ):
                    continue

                if self.mode == "clean_stopwords":

                    word = token.text.lower().strip()

                elif self.mode == "full_lemma":

                    word = token.lemma_.lower().strip()

                else:

                    raise ValueError(
                        f"Unknown normalization mode: {self.mode}"
                    )

                if len(word) > 1 and word != "-PRON-":
                    tokens.append(word)

            output.append(" ".join(tokens))

        return output


def get_preprocessing_configs():

    return {
        "norm_1_baseline": TextNormalizer(
            mode="baseline"
        ),

        "norm_2_stopwords": TextNormalizer(
            mode="clean_stopwords"
        ),

        "norm_3_lemma": TextNormalizer(
            mode="full_lemma"
        ),
    }


def get_vectorizer_configs():

    return {

        "vec_binary_1": CountVectorizer(
            binary=True,
            ngram_range=(1, 1),
            min_df=2
        ),

        "vec_freq_1_2": CountVectorizer(
            binary=False,
            ngram_range=(1, 2),
            min_df=2
        ),

        "vec_tfidf_1": TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=2
        ),

        "vec_tfidf_1_2": TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2
        ),
    }


def build_pipeline(
    normalizer,
    vectorizer,
    estimator,
    cache_dir="results/pipeline_cache"
):

    """
    Build a complete Pipeline.

    All learned transformations are fitted only on
    the training partition.
    """

    Path(cache_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    memory = Memory(
        location=cache_dir,
        verbose=0
    )

    return Pipeline(
        steps=[
            ("normalizer", clone(normalizer)),
            ("vectorizer", clone(vectorizer)),
            ("model", clone(estimator)),
        ],
        memory=memory
    )
