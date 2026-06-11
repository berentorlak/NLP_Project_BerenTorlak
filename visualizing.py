# Part where we have the visualizations.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


class Visualizer:
    """
    Class for visualizing alignment and clustering results.

    """

    def __init__(self, output_dir: str = "figures"): # directory to save figures
       
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    # 1. Alignment scores over time

    def plot_scores_by_year(self, summary: dict) -> None:
        """
        Line plot of mean alignment score per year with std band.

        """
        years = sorted(summary.keys())
        means = [summary[y]["mean"] for y in years]
        stds  = [summary[y]["std"]  for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, means, marker="o", linewidth=2,
                color="#2196F3", label="Mean score")
        ax.fill_between(
            years,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2, color="#2196F3", label="± Std"
        )
        ax.set_title("Thematic Alignment Score of JBI Articles Over Time", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{self.output_dir}/1_scores_by_year.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")


    # 2. Score distribution

    def plot_score_distribution(self, scores: np.ndarray) -> None:
        """
        Histogram of all alignment scores with mean line.

        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins=40, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax.axvline(scores.mean(), color="red", linewidth=2,
                   linestyle="--", label=f"Mean: {scores.mean():.3f}")
        ax.axvline(np.percentile(scores, 10), color="orange", linewidth=1.5,
                   linestyle=":", label=f"10th percentile: {np.percentile(scores, 10):.3f}")
        ax.set_title("Distribution of Thematic Alignment Scores", fontsize=14)
        ax.set_xlabel("Cosine Similarity Score")
        ax.set_ylabel("Number of Articles")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{self.output_dir}/2_score_distribution.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")

    # 3. 2D article map

    def plot_2d_map(self, coords: np.ndarray, labels: np.ndarray,
                    cluster_names: dict, scores: np.ndarray) -> None:
        """
        2D scatter plot of articles colored by cluster.
        Outliers (score < 0.1) are marked separately.

        Parameters;
        coords: np.ndarray — 2D coordinates from Clusterer.get_2d_coords()
        labels: np.ndarray — cluster assignments
        cluster_names : dict — output of Clusterer.get_cluster_names()
        scores: np.ndarray — alignment scores
        """
        colors = ["#2196F3", "#4CAF50"]
        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster_id, info in cluster_names.items():
            mask = labels == cluster_id
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[cluster_id % len(colors)],
                label=info["label"],
                alpha=0.5, s=15,
            )

        # Marking outliers wit X
        outlier_mask = scores < 0.1
        ax.scatter(
            coords[outlier_mask, 0], coords[outlier_mask, 1],
            c="black", marker="x", s=40, linewidths=1.5,
            label="Outliers (score < 0.1)", zorder=5,
        )

        ax.set_title("2D Semantic Map of JBI Articles by Cluster", fontsize=14)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = f"{self.output_dir}/3_2d_article_map.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")

    # 4. Cluster distribution by year

    def plot_cluster_distribution_by_year(self, yearly_dist: dict,
                                           cluster_names: dict) -> None:
        """
        Stacked bar chart showing how cluster proportions change over time.

        """
        colors = [ "#FF9800", "#9C27B0"]
        years  = sorted(yearly_dist.keys())
        n_clusters = len(cluster_names)

        # Normalizing to percentages
        proportions = {i: [] for i in range(n_clusters)}
        for year in years:
            total = sum(yearly_dist[year].values())
            for i in range(n_clusters):
                proportions[i].append(yearly_dist[year][i] / total * 100)

        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(years))

        for i in range(n_clusters):
            ax.bar(years, proportions[i], bottom=bottom,
                   color=colors[i % len(colors)],
                   label=cluster_names[i]["label"], alpha=0.85)
            bottom += np.array(proportions[i])

        ax.set_title("Cluster Distribution by Year (%)", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion of Articles (%)")
        ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1, 1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        path = f"{self.output_dir}/4_cluster_distribution_by_year.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")


    # 5. Mean alignment score for both clusters

    def plot_cluster_alignment_scores(self, articles: list, scores: np.ndarray,
                                       labels: np.ndarray, cluster_names: dict) -> None:
        """
        Bar chart of mean alignment score for both clusters.

        Parameters;
        articles: list — article list
        scores: np.ndarray — alignment scores
        labels: np.ndarray — cluster assignments
        cluster_names : dict — output of Clusterer.get_cluster_names()
        """
        colors      = ["#FF9800", "#9C27B0"]
        mean_scores = []
        std_scores  = []
        names       = []

        for i in range(len(cluster_names)):
            mask = labels == i
            mean_scores.append(scores[mask].mean())
            std_scores.append(scores[mask].std())
            names.append(cluster_names[i]["label"])

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, mean_scores, color=colors[:len(cluster_names)],
                      alpha=0.85, yerr=std_scores, capsize=5)
        ax.set_title("Mean Alignment Score by Cluster", fontsize=14)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = f"{self.output_dir}/5_cluster_alignment_scores.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")

    # 6. Outliers by year

    def plot_outliers_by_year(self, articles: list, scores: np.ndarray,
                               threshold: float = 0.1) -> None:
        """
        Bar chart showing number of outlier articles per year.

        """
        outlier_counts = {}
        total_counts   = {}

        for i, article in enumerate(articles):
            year = article["year"]
            total_counts[year]   = total_counts.get(year, 0) + 1
            if scores[i] < threshold:
                outlier_counts[year] = outlier_counts.get(year, 0) + 1

        years          = sorted(total_counts.keys())
        outlier_ratios = [outlier_counts.get(y, 0) / total_counts[y] * 100
                          for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(years, outlier_ratios, color="#F44336", alpha=0.8)
        ax.set_title(f"Percentage of Outlier Articles per Year (score < {threshold})",
                     fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Outlier Articles (%)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = f"{self.output_dir}/6_outliers_by_year.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")

    # 7. High alignment percentage by year

    def plot_high_alignment_ratio(self, articles: list, scores: np.ndarray,
                                   threshold: float = 0.4) -> None:
        """
        Line plot of percentage of highly aligned articles per year.

        """
        high_counts  = {}
        total_counts = {}

        for i, article in enumerate(articles):
            year = article["year"]
            total_counts[year] = total_counts.get(year, 0) + 1
            if scores[i] >= threshold:
                high_counts[year] = high_counts.get(year, 0) + 1

        years  = sorted(total_counts.keys())
        ratios = [high_counts.get(y, 0) / total_counts[y] * 100 for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, ratios, marker="o", linewidth=2, color="#9C27B0")
        ax.fill_between(years, ratios, alpha=0.2, color="#9C27B0")
        ax.set_title(f"Percentage of Highly Aligned Articles per Year (score ≥ {threshold})",
                     fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Highly Aligned Articles (%)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{self.output_dir}/7_high_alignment_ratio.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")

   
    # 8. Cluster alignment scores over time
   
    def plot_cluster_scores_over_time(self, articles: list, scores: np.ndarray,
                                       labels: np.ndarray, cluster_names: dict) -> None:
        """
        Line plot of mean alignment score per cluster per year.

        """
        colors = ["#FF9800", "#9C27B0"]

        # Grouping scores by cluster and year
        data = {i: {} for i in range(len(cluster_names))}
        for idx, article in enumerate(articles):
            year       = article["year"]
            cluster_id = labels[idx]
            if year not in data[cluster_id]:
                data[cluster_id][year] = []
            data[cluster_id][year].append(scores[idx])

        years  = sorted(set(a["year"] for a in articles))
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, info in cluster_names.items():
            means = [np.mean(data[i][y]) if y in data[i] else None for y in years]
            ax.plot(years, means, marker="o", linewidth=2,
                    color=colors[i % len(colors)], label=info["label"])

        ax.set_title("Mean Alignment Score per Cluster Over Time", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=9, bbox_to_anchor=(1, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{self.output_dir}/8_cluster_scores_over_time.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[visualizer] Saved → {path}")


# 9. Donut chart grid by year

    def plot_donut_grid(self, yearly_dist: dict, cluster_names: dict) -> None:
        """
        Grid of donut charts showing thematic composition by year. 

        """
        colors = ["#2196F3", "#9C27B0"]
        years  = sorted(yearly_dist.keys())
        n_cols = 5
        n_rows = int(np.ceil(len(years) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9 * n_rows))
        axes = axes.flatten()

        for idx, year in enumerate(years):
            ax     = axes[idx]
            counts = [yearly_dist[year][i] for i in range(len(cluster_names))]
            total  = sum(counts)
            labels = [
                f"{cluster_names[i]['label']}\n{counts[i]/total*100:.1f}%"
                for i in range(len(cluster_names))
            ]

            wedges, _ = ax.pie(
                counts,
                colors=colors[:len(cluster_names)],
                startangle=90,
                wedgeprops=dict(width=0.5),   # thickness of donuts
            )

            ax.set_title(f"{year}\n(n={total})", fontsize=11, fontweight="bold")

            # percentages inside
            for i, (wedge, count) in enumerate(zip(wedges, counts)):
                angle  = (wedge.theta2 + wedge.theta1) / 2
                x      = 0.7 * np.cos(np.radians(angle))
                y      = 0.7 * np.sin(np.radians(angle))
                ax.text(x, y, f"{count/total*100:.1f}%",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")

        # In case we have less or more years later, hide empties
        for idx in range(len(years), len(axes)):
            axes[idx].set_visible(False)


        legend_labels = [cluster_names[i]["label"] for i in range(len(cluster_names))]
        legend_patches = [
            plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i])
            for i in range(len(cluster_names))
        ]
        fig.legend(handles=legend_patches, loc="lower center",
                ncol=len(cluster_names), fontsize=10,
                bbox_to_anchor=(0.5, -0.02))

        fig.suptitle("Thematic Composition by Year", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 0.97], h_pad=14.0)
        path = f"{self.output_dir}/9_donut_grid.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[visualizer] Saved → {path}")


# Test
if __name__ == "__main__":
    import json
    from evaluating import Evaluator
    from clustering import Clusterer

    # Loading and computing everything
    evaluator = Evaluator()
    evaluator.load("data/abstracts.json", "data/embeddings.npz")
    evaluator.compute_scores()
    summary = evaluator.results_by_year()

    clusterer = Clusterer()
    clusterer.load("data/abstracts.json", "data/embeddings.npz")
    clusterer.fit()
    cluster_names = clusterer.get_cluster_names()
    coords = clusterer.get_2d_coords(method="umap")
    labels        = clusterer.get_labels()
    yearly_dist   = clusterer.get_yearly_distribution()

    # Showing the plots
    viz = Visualizer()
    viz.plot_scores_by_year(summary)
    viz.plot_score_distribution(evaluator.scores)
    viz.plot_2d_map(coords, labels, cluster_names, evaluator.scores)
    viz.plot_cluster_distribution_by_year(yearly_dist, cluster_names)
    viz.plot_cluster_alignment_scores(evaluator.articles, evaluator.scores, labels, cluster_names)
    viz.plot_outliers_by_year(evaluator.articles, evaluator.scores)
    viz.plot_high_alignment_ratio(evaluator.articles, evaluator.scores)
    viz.plot_cluster_scores_over_time(evaluator.articles, evaluator.scores, labels, cluster_names)
    viz.plot_donut_grid(yearly_dist, cluster_names)