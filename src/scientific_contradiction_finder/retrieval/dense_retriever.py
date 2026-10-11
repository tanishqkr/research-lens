"""Dense bi-encoder retriever using local sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

from scientific_contradiction_finder.schemas import Paper


class DenseRetriever:
    """Bi-encoder dense semantic retriever."""

    def __init__(
        self,
        papers: list[Paper],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.papers = papers
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        self.doc_embeddings: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def index(self):
        """Precomputes embeddings for the loaded papers."""
        if not self.papers:
            return
        texts = [f"{p.title}. {p.abstract}" for p in self.papers]
        self.doc_embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[Paper, float]]:
        if not self.papers:
            return []
        if self.doc_embeddings is None:
            self.index()

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        # Cosine similarity since vectors are L2-normalized
        scores = np.dot(self.doc_embeddings, query_embedding)

        scored = [(self.papers[i], float(scores[i])) for i in range(len(self.papers))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
