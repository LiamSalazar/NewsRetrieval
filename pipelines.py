import re
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="baseline"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._process_text(text) for text in X]

    def _process_text(self, text):
        text = str(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if self.mode == "baseline":
            return text.lower()

        text_clean = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]', ' ', text)
        doc = nlp(text_clean)

        stop_pos = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "PUNCT", "SPACE"}

        tokens = []
        for token in doc:
            if token.is_space or token.pos_ in stop_pos:
                continue
            if self.mode == "clean_stopwords":
                word = token.text.lower().strip()
            elif self.mode == "full_lemma":
                word = token.lemma_.lower().strip()
            else:
                word = token.text.lower().strip()

            if len(word) > 1:
                tokens.append(word)

        return " ".join(tokens)


def get_preprocessing_configs():
    """Retorna las 3 configuraciones obligatorias de normalización."""
    return {
        "norm_1_baseline": TextNormalizer(mode="baseline"),
        "norm_2_stopwords": TextNormalizer(mode="clean_stopwords"),
        "norm_3_lemma": TextNormalizer(mode="full_lemma"),
    }


def get_vectorizer_configs():
    """Retorna configuraciones de representación: binaria, frecuencia y TF-IDF con n-gramas"""
    return {
        "vec_binary_1": CountVectorizer(binary=True, ngram_range=(1, 1), min_df=2),
        "vec_freq_1_2": CountVectorizer(binary=False, ngram_range=(1, 2), min_df=2),
        "vec_tfidf_1": TfidfVectorizer(ngram_range=(1, 1), min_df=2),
        "vec_tfidf_1_2": TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    }


if __name__ == "__main__":
    print("Cargando labeled_documents.tsv")
    df = pd.read_csv("labeled_documents.tsv", sep="\t")
    train_texts = df[df["split"] == "train"]["text"].tolist()
    dev_texts = df[df["split"] == "dev"]["text"].tolist()

    sample_text = train_texts[0]
    print(f"\nTexto original: {sample_text[:120]}...\n")

    prep_configs = get_preprocessing_configs()
    vec_configs = get_vectorizer_configs()

    for norm_name, normalizer in prep_configs.items():
        processed_sample = normalizer.transform([sample_text])[0]
        print(f"[{norm_name}]: {processed_sample[:100]}...")

        pipe = Pipeline([
            ("norm", normalizer),
            ("vec", vec_configs["vec_tfidf_1"])
        ])
        X_train = pipe.fit_transform(train_texts)
        X_dev = pipe.transform(dev_texts)
        print(f"   Matriz train: {X_train.shape} | Matriz dev: {X_dev.shape}\n")