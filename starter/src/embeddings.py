"""Local embedding adapter backed by Chroma's ONNX default model."""
from typing import Protocol

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class ChromaLocalEmbedder:
    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions = 384
    max_input_tokens = 256

    def __init__(self, model_name: str = default_model) -> None:
        if model_name != self.default_model:
            raise ValueError(
                f"ChromaLocalEmbedder supports only {self.default_model!r}; "
                "implement another adapter to use a different model"
            )
        self.model_name = f"{model_name} (Chroma ONNX)"
        self._embedding_function = DefaultEmbeddingFunction()
        self._tokenizer = ONNXMiniLM_L6_V2().tokenizer
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._embedding_function(texts)
        vectors = [[float(value) for value in embedding] for embedding in embeddings]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(f"{self.model_name} returned a vector with an unexpected dimension")
        return vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = self.embed_documents([question])
        return vectors[0]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
