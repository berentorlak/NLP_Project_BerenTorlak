# Embedding article abstracts and the journal Aims & Scope into vector representations
# using a Sentence‑Transformers model.

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2" #default

# Journal's Aims & Scope paragraph 
AIMS_AND_SCOPE = """
The Journal of Biomedical Informatics (JBI) is the premier methodology journal 
in the field of biomedical informatics. JBI publishes research on new methodologies 
and techniques that have general applicability and form the basis for the evolving 
science of biomedical informatics. JBI seeks to publish papers that make a conceptual 
contribution to the field, typically by describing an innovation in methodology or 
technique or by discussing substantive generalizable lessons that have been learned 
in the context of an informatics project. Papers must build on deep understanding 
and utilization of medical domain knowledge and should consider pragmatic translation 
for clinical care or applications.
"""

class Embedder: #text to vectors


    def __init__(self, model_name: str = MODEL_NAME):
        """
        Parameters;
        model_name : str 
        """
        print(f"[embedder] Model is loading: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.article_embeddings = None  # vectors of the articles
        self.scope_embedding    = None  # aims & scope vector
        self.articles           = None  # original articles list

    def embed_articles(self, abstracts_path: str) -> np.ndarray:
        """
        Embeds the abstracts of articles.

        Parameters;
        abstracts_path : str — path to the abstracts.json file

        Returns;
        np.ndarray
        """
        # Dowloading the JSON file
        with open(abstracts_path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)

        # only the abstracts
        texts = [a["abstract"] for a in self.articles]
        print(f"[embedder] {len(texts)} articles with abstracts are being embedded...")

    
        self.article_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,  
            batch_size=64,           
        )

        print(f"[embedder] Completed. Dimension: {self.article_embeddings.shape}")
        return self.article_embeddings

    def embed_scope(self, scope_text: str = AIMS_AND_SCOPE) -> np.ndarray:
        """
        Embeds the journal's Aims & Scope paragraph.

        Parameters;
        scope_text : str — aims & scope text

        Returns;
        np.ndarray — single vector
        """
        print("[embedder] Aims & Scope vector is being created...")
        self.scope_embedding = self.model.encode([scope_text])[0]
        print(f"[embedder] Completed. Dimension: {self.scope_embedding.shape}")
        return self.scope_embedding

    def save(self, path: str) -> None:
        """
        Saves the vectors to a file.

        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            article_embeddings=self.article_embeddings,
            scope_embedding=self.scope_embedding,
        )
        print(f"[embedder] Saved → {path}")

    def load(self, path: str) -> None:
        """
        Loads previously saved vectors.

        """
        data = np.load(path)
        self.article_embeddings = data["article_embeddings"]
        self.scope_embedding    = data["scope_embedding"]
        print(f"[embedder] Loaded → {path}")
        print(f"[embedder] Number of articles: {len(self.article_embeddings)}")


# Test
if __name__ == "__main__":
    embedder = Embedder()
    embedder.embed_articles("data/abstracts.json")
    embedder.embed_scope()
    embedder.save("data/embeddings.npz")