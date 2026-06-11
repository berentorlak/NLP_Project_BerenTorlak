# Part where we group articles into thematic clusters.

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


# Words that are too common in JBI to be distinctive
JBI_STOPWORDS = [
    "data", "based", "model", "models", "information", "clinical",
    "study", "results", "method", "methods", "approach", "using",
    "used", "proposed", "paper", "work", "performance", "system"
]

# Manual cluster labels (chosed based on the top TF-IDF keywords and abstracts)
MANUAL_LABELS = {
    0: "Clinical & Health Informatics",
    1: "Biomedical NLP & Machine Learning",
}


class Clusterer:
    """
    Class that groups articles into thematic clusters.

    """

    def __init__(self):
        self.n_clusters    = None
        self.articles      = None
        self.embeddings    = None
        self.kmeans        = None
        self.labels        = None
        self.coords_2d     = None
        self.cluster_names = None

    def load(self, abstracts_path: str, embeddings_path: str) -> None:
        """
        Loads article list and embeddings.

        """
        with open(abstracts_path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)

        data = np.load(embeddings_path)
        self.embeddings = data["article_embeddings"]
        print(f"[clusterer] {len(self.articles)} articles loaded")

    def find_optimal_clusters(self, max_k: int = 10) -> int:
        """
        Finds the optimal number of clusters using Silhouette Score.
     
        """
        print("[clusterer] Finding optimal number of clusters (Silhouette)...")
        scores = {}

        for k in range(2, max_k + 1):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.embeddings)
            score  = silhouette_score(self.embeddings, labels, sample_size=500)
            scores[k] = score
            print(f"  k={k}: silhouette score={score:.4f}")

        optimal_k = max(scores, key=scores.get)
        print(f"[clusterer] Optimal number of clusters: {optimal_k}")
        return optimal_k

    def fit(self) -> None:
        """
        Finds optimal k automatically then runs K-Means.
        """
        self.n_clusters = self.find_optimal_clusters(max_k=10)

        print(f"[clusterer] Creating {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        self.labels = self.kmeans.fit_predict(self.embeddings)

        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"[clusterer] Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  [{cluster_id}]: {count} articles")

    def get_cluster_names(self) -> dict:
        """
        Finds top TF-IDF keywords per cluster for reference.
        Then assigns cluster names using manual labels defined in MANUAL_LABELS based on 
        top keywords.
        The manual names were defined after checking the top keywords!!
        """
        print("[clusterer] Assigning cluster names...")

        cluster_texts = {i: [] for i in range(self.n_clusters)}
        for i, article in enumerate(self.articles):
            cluster_texts[self.labels[i]].append(
                article["title"] + " " + article["abstract"]
            )

        corpus = [" ".join(cluster_texts[i]) for i in range(self.n_clusters)]

        base_stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words())
        all_stop_words  = base_stop_words + JBI_STOPWORDS 
            # we are skipping them because they are not distinctive


        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=all_stop_words,
            ngram_range=(1, 2),
        )
        tfidf_matrix  = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        self.cluster_names = {}
        for i in range(self.n_clusters):
            top_indices = tfidf_matrix[i].toarray().flatten().argsort()[-5:][::-1]
            top_words   = [feature_names[j] for j in top_indices]
            label       = MANUAL_LABELS.get(i, " / ".join(top_words[:3]))
            self.cluster_names[i] = {
                "label"   : label,
                "keywords": top_words,
            }
            print(f"  Cluster {i} ({', '.join(top_words[:3])}) → {label}")

        return self.cluster_names

    def get_2d_coords(self, method: str = "umap") -> np.ndarray:
        """
        Reduces embeddings to 2D for visualization.
        Uses UMAP.

        """
        print(f"[clusterer] Computing 2D coordinates using {method.upper()}...")

        if method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
            self.coords_2d = reducer.fit_transform(self.embeddings)
        else:
            from sklearn.decomposition import PCA # in case we want to use PCA for something
            pca            = PCA(n_components=2, random_state=42)
            self.coords_2d = pca.fit_transform(self.embeddings)

        print("[clusterer] Done")
        return self.coords_2d

    def get_labels(self) -> np.ndarray:
        """Returns the cluster number assigned to each article."""
        return self.labels

    def get_yearly_distribution(self) -> dict:
        """
        Computes cluster distribution for evert year.

        """
        distribution = {}
        for i, article in enumerate(self.articles):
            year       = article["year"]
            cluster_id = self.labels[i]
            if year not in distribution:
                distribution[year] = {j: 0 for j in range(self.n_clusters)}
            distribution[year][cluster_id] += 1
        return distribution


# Test
if __name__ == "__main__":
    clusterer = Clusterer()
    clusterer.load("data/abstracts.json", "data/embeddings.npz")
    clusterer.fit()
    names  = clusterer.get_cluster_names()
    coords = clusterer.get_2d_coords(method="umap")

    print("\nCluster summary:") # to show a more detailed top words
    for cluster_id, info in names.items():
        print(f"  Cluster {cluster_id} ({', '.join(info['keywords'])}) → {info['label']}")