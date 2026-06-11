#The part we measure how well each article's abstract matches the journal's 
# Aims & Scope text by calculating cosine similarity between their vector representations.

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:
    """
    Class that calculates the scores.

    """

    def __init__(self):
        self.articles           = None   
        self.article_embeddings = None   # article vectors
        self.scope_embedding    = None   # aims & scope vector
        self.scores             = None   

    def load(self, abstracts_path: str, embeddings_path: str) -> None:
        """
        Loads the list of articles and their embeddings.

        """
        with open(abstracts_path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)

        data = np.load(embeddings_path)
        self.article_embeddings = data["article_embeddings"]
        self.scope_embedding    = data["scope_embedding"]

        print(f"[evaluator] {len(self.articles)} articles loaded.")

    def compute_scores(self) -> np.ndarray:
        """
        Computes cosine similarity scores for each article 
        against the journal's Aims & Scope.
        Returns;
        np.ndarray — score between 0 and 1 for each article
        """
        # Scope vector to matrix
        scope = self.scope_embedding.reshape(1, -1)

        # The score of similarity between each article and the scope
        self.scores = cosine_similarity(self.article_embeddings, scope).flatten()

        print(f"[evaluator] Scores are calculated.")
        print(f"[evaluator] Mean score : {self.scores.mean():.3f}")
        print(f"[evaluator] Highest score: {self.scores.max():.3f}")
        print(f"[evaluator] Lowest score : {self.scores.min():.3f}")

        return self.scores

    def results_by_year(self) -> dict:
        """
        Groups the scores by year and generates summary statistics for each year.

        """
        results = {}

       
        for i, article in enumerate(self.articles):
            year = article["year"]
            if year not in results:
                results[year] = []
            results[year].append(self.scores[i])

        # Statistics for every year
        summary = {}
        for year in sorted(results.keys()):
            year_scores = np.array(results[year])
            summary[year] = {
                "mean"  : round(float(year_scores.mean()),   3),
                "median": round(float(np.median(year_scores)), 3),
                "std"   : round(float(year_scores.std()),    3),
                "count" : len(year_scores),
            }

        return summary

    def top_articles(self, n: int = 5) -> list:
        """
        Returns the articles with the highest scores.

        Parameters;
        n : int — number of articles to return

        """
        top_indices = self.scores.argsort()[-n:][::-1]
        return [
            {
                "rank" : i + 1,
                "score": round(float(self.scores[idx]), 3),
                "title": self.articles[idx]["title"],
                "year" : self.articles[idx]["year"],
            }
            for i, idx in enumerate(top_indices)
        ]

    def bottom_articles(self, n: int = 5) -> list:
        """
        Returns the articles with the lowest scores.

        Parameters;
        n : int — number of articles to return

        """
        bottom_indices = self.scores.argsort()[:n]
        return [
            {
                "rank" : i + 1,
                "score": round(float(self.scores[idx]), 3),
                "title": self.articles[idx]["title"],
                "year" : self.articles[idx]["year"],
            }
            for i, idx in enumerate(bottom_indices)
        ]


# Test
if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.load("data/abstracts.json", "data/embeddings.npz")
    evaluator.compute_scores()

    print("\nAverage alignment score by year:")
    summary = evaluator.results_by_year()
    for year, stats in summary.items():
        print(f"  {year}: mean={stats['mean']}  number of articles={stats['count']}")

    print("\n5 most aligned articles:")
    for a in evaluator.top_articles(5):
        print(f"  {a['rank']}. [{a['score']}] {a['title'][:70]}")

    print("\n5 least aligned articles:")
    for a in evaluator.bottom_articles(5):
        print(f"  {a['rank']}. [{a['score']}] {a['title'][:70]}")