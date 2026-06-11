# Exploratory Data Analysis module 

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


class EDA:

    def __init__(self, output_dir: str = "figures"):
        """
        Parameters;
        output_dir : str — place to save figures
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.articles   = None

    def load(self, abstracts_path: str) -> None:
        """
        Loads the article corpus.

        Parameters;
        abstracts_path : str — path to abstracts.json
        """
        with open(abstracts_path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)
        print(f"[eda] {len(self.articles)} articles loaded")

    def summary(self) -> None:
        """
        Prints basic corpus statistics.
        """
        years    = [a["year"] for a in self.articles]
        lengths  = [len(a["abstract"].split()) for a in self.articles]

        print(f"\n--- Corpus Summary ---")
        print(f"Total articles      : {len(self.articles)}")
        print(f"Year range          : {min(years)} - {max(years)}")
        print(f"Mean abstract length: {np.mean(lengths):.0f} words")
        print(f"Min abstract length : {min(lengths)} words")
        print(f"Max abstract length : {max(lengths)} words")
        print(f"----------------------\n")

  
    # Articles per year
   
    def plot_articles_per_year(self) -> None:
        """
        Bar chart of number of articles published every year.
        """
        year_counts = Counter(a["year"] for a in self.articles)
        years  = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(years, counts, color="#2196F3", alpha=0.85)
        ax.set_title("Number of JBI Articles per Year", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Articles")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(True, alpha=0.3, axis="y")

      
        for year, count in zip(years, counts):
            ax.text(year, count + 1, str(count),
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = f"{self.output_dir}/eda_1_articles_per_year.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[eda] Saved → {path}")

 
    # Abstract length distribution

    def plot_abstract_length_distribution(self) -> None:
        """
        Histogram of abstract lengths in words.
        """
        lengths = [len(a["abstract"].split()) for a in self.articles]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(lengths, bins=40, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(lengths), color="red", linewidth=2,
                   linestyle="--", label=f"Mean: {np.mean(lengths):.0f} words")
        ax.set_title("Abstract Length Distribution", fontsize=14)
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Number of Articles")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.output_dir}/eda_2_abstract_lengths.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[eda] Saved → {path}")

    # Top keywords across entire corpus
    
    def plot_top_keywords(self, n: int = 20) -> None:
        """
        Horizontal bar chart of the most frequent keywords in the corpus.
        It uses TF-IDF to find distinctive terms.

        Parameters;
        n : int — number of top keywords to show
        """
        # Combining all abstracts
        all_texts = [a["title"] + " " + a["abstract"] for a in self.articles]

        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Mean TF-IDF score across all documents
        mean_scores = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[-n:][::-1]
        top_words   = [feature_names[i] for i in top_indices]
        top_scores  = [mean_scores[i]   for i in top_indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_words[::-1], top_scores[::-1], color="#FF9800", alpha=0.85)
        ax.set_title(f"Top {n} Keywords in JBI Corpus (TF-IDF)", fontsize=14)
        ax.set_xlabel("Mean TF-IDF Score")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        path = f"{self.output_dir}/eda_3_top_keywords.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[eda] Saved → {path}")

    # Keyword trends over time
  
    def plot_keyword_trends(self, keywords: list = None) -> None:
        """
        Line plot tracking how often specific keywords appear every year.

        Parameters;
        keywords : list — keywords to track 
        """
        if keywords is None:
            keywords = [
                "deep learning",
                "machine learning",
                "neural network",
                "natural language processing",
                "electronic health record",
            ]

        # Counting keyword occurrences evry year
        year_counts = {kw: {} for kw in keywords}
        total_per_year = Counter(a["year"] for a in self.articles)

        for article in self.articles:
            year = article["year"]
            text = (article["title"] + " " + article["abstract"]).lower()
            for kw in keywords:
                if kw in text:
                    year_counts[kw][year] = year_counts[kw].get(year, 0) + 1

        years  = sorted(total_per_year.keys())
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, kw in enumerate(keywords):
            # Normalized by total articles per year
            ratios = [
                year_counts[kw].get(y, 0) / total_per_year[y] * 100
                for y in years
            ]
            ax.plot(years, ratios, marker="o", linewidth=2,
                    color=colors[i % len(colors)], label=kw)

        ax.set_title("Keyword Trends Over Time (% of Articles per Year)", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage of Articles (%)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.output_dir}/eda_4_keyword_trends.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[eda] Saved → {path}")


# Test
if __name__ == "__main__":
    eda = EDA()
    eda.load("data/abstracts.json")
    eda.summary()
    eda.plot_articles_per_year()
    eda.plot_abstract_length_distribution()
    eda.plot_top_keywords()
    eda.plot_keyword_trends()