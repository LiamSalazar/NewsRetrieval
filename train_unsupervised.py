import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pipelines import get_preprocessing_configs


def run_unsupervised_experiments():
    df = pd.read_csv("labeled_documents.tsv", sep="\t")
    
    texts = df["text"].tolist()
    y_true = df["category"].values

    prep_configs = get_preprocessing_configs()

    normalizer = prep_configs["norm_3_lemma"]
    processed_texts = normalizer.transform(texts)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(processed_texts)

    experiments_log = []
 
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    clusters_kmeans = kmeans.fit_predict(X_tfidf)
    
    ari_km = adjusted_rand_score(y_true, clusters_kmeans)
    nmi_km = normalized_mutual_info_score(y_true, clusters_kmeans)
    
    print(f"K-Means (k=3), ARI: {ari_km:.4f}, NMI: {nmi_km:.4f}")
    experiments_log.append({
        "experiment_id": "EXP_B_001",
        "task": "clustering",
        "normalization": "norm_3_lemma",
        "representation": "tfidf_5000",
        "method": "KMeans",
        "hyperparameters": "n_clusters=3",
        "dev_metric": round(ari_km, 4),
        "selected": True
    })

    agg = AgglomerativeClustering(n_clusters=3)
    clusters_agg = agg.fit_predict(X_tfidf.toarray())
    
    ari_agg = adjusted_rand_score(y_true, clusters_agg)
    nmi_agg = normalized_mutual_info_score(y_true, clusters_agg)
    
    print(f"AgglomerativeClustering, ARI: {ari_agg:.4f}, NMI: {nmi_agg:.4f}")
    experiments_log.append({
        "experiment_id": "EXP_B_002",
        "task": "clustering",
        "normalization": "norm_3_lemma",
        "representation": "tfidf_5000",
        "method": "AgglomerativeClustering",
        "hyperparameters": "n_clusters=3",
        "dev_metric": round(ari_agg, 4),
        "selected": False
    })

    
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 1))
    X_counts = count_vectorizer.fit_transform(processed_texts)
    
    lda = LatentDirichletAllocation(n_components=3, random_state=0)
    lda.fit(X_counts)
    
    print("\nTop words per topic (LDA):")
    feature_names_lda = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names_lda[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"   - Topic {topic_idx + 1}: {', '.join(top_words)}")

    nmf = NMF(n_components=3, random_state=0)
    nmf.fit(X_tfidf)
    
    print("\nTop words per topic (NMF):")
    feature_names_tfidf = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names_tfidf[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    experiments_log.append({
        "experiment_id": "EXP_B_003",
        "task": "topic_modeling",
        "normalization": "norm_3_lemma",
        "representation": "count_5000",
        "method": "LDA",
        "hyperparameters": "n_components=3",
        "dev_metric": 0.0,
        "selected": True
    })
    experiments_log.append({
        "experiment_id": "EXP_B_004",
        "task": "topic_modeling",
        "normalization": "norm_3_lemma",
        "representation": "tfidf_5000",
        "method": "NMF",
        "hyperparameters": "n_components=3",
        "dev_metric": 0.0,
        "selected": False
    })

    try:
        existing_exp = pd.read_csv("experiments.tsv", sep="\t")
        combined_exp = pd.concat([existing_exp, pd.DataFrame(experiments_log)], ignore_index=True)
    except FileNotFoundError:
        combined_exp = pd.DataFrame(experiments_log)
        
    combined_exp.to_csv("experiments.tsv", sep="\t", index=False)


if __name__ == "__main__":
    run_unsupervised_experiments()