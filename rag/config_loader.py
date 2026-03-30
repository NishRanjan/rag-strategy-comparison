"""Build LangChain components from YAML experiment configs."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    from langchain_core.documents import Document
except Exception:  # noqa: BLE001
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)


class HeuristicLLM:
    """Tiny deterministic LLM stand-in for tests and offline use."""

    def invoke(self, prompt_value):
        return _heuristic_response(prompt_value)


class InMemoryVectorStore:
    """Minimal vector store fallback when Chroma is unavailable."""

    def __init__(self, documents: list[Document], embeddings) -> None:
        self._documents = documents
        self._embeddings = embeddings
        self._vectors = embeddings.embed_documents([doc.page_content for doc in documents])

    def as_retriever(self, search_kwargs: dict | None = None):
        k = int((search_kwargs or {}).get("k", 5))
        return DenseRetriever(self._documents, self._vectors, self._embeddings, k=k)


class DenseRetriever:
    def __init__(self, documents: list[Document], vectors: list[list[float]], embeddings, k: int = 5):
        self._documents = documents
        self._vectors = vectors
        self._embeddings = embeddings
        self._k = k

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(l * r for l, r in zip(left, right))

    def invoke(self, query: str) -> list[Document]:
        query_vector = self._embeddings.embed_query(query)
        ranked = sorted(
            zip(self._documents, self._vectors, strict=False),
            key=lambda item: self._dot(query_vector, item[1]),
            reverse=True,
        )
        return [doc for doc, _vector in ranked[: self._k]]


class SparseRetriever:
    def __init__(self, documents: list[Document], k: int = 5) -> None:
        self._documents = documents
        self._k = k

    def invoke(self, query: str) -> list[Document]:
        query_tokens = set(query.lower().split())
        ranked = sorted(
            self._documents,
            key=lambda doc: len(query_tokens & set(doc.page_content.lower().split())),
            reverse=True,
        )
        return ranked[: self._k]


class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5, k: int = 5) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._alpha = alpha
        self._k = k

    def invoke(self, query: str) -> list[Document]:
        scores: dict[int, float] = {}
        docs_by_id: dict[int, Document] = {}
        for weight, retriever in ((self._alpha, self._dense), (1 - self._alpha, self._sparse)):
            for rank, doc in enumerate(retriever.invoke(query), start=1):
                doc_id = id(doc)
                docs_by_id[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / rank)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [docs_by_id[doc_id] for doc_id, _score in ranked[: self._k]]


class HashEmbeddings:
    """Small deterministic embedding model for local tests and offline runs."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dimensions
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign
        scale = float(len(tokens))
        return [value / scale for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


class GeminiEmbeddings:
    """Use Gemini task types tuned separately for documents and queries."""

    def __init__(self, model: str) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._document_embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            task_type="retrieval_document",
        )
        self._query_embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            task_type="retrieval_query",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._document_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._query_embeddings.embed_query(text)


def _heuristic_response(prompt_value) -> str:
    """Return a deterministic non-empty response for local tests."""
    text = getattr(prompt_value, "to_string", lambda: str(prompt_value))()
    lowered = text.lower()
    if "rewrite the user's question" in lowered:
        marker = "human:"
        return text.split(marker)[-1].strip() if marker in lowered else text.strip()
    if "write a short hypothetical document" in lowered:
        return "This product information answers the user question with a likely supporting detail."
    if "reply with only 'yes' or 'no'" in lowered:
        return "yes"
    if "question:" in text:
        question = text.rsplit("Question:", 1)[-1].strip()
        return f"Based on the provided context, the answer to '{question}' is in the retrieved documents."
    return "Based on the provided context, the answer is in the retrieved documents."


def build_llm(llm_config: dict):
    """Build a LangChain chat model from a provider/model config dict.

    Config keys: provider (gemini | azure_openai | anthropic | heuristic), model,
    temperature, max_tokens.
    """
    provider = llm_config["provider"]
    model = llm_config["model"]
    kwargs = {k: v for k, v in llm_config.items() if k not in {"provider", "model"}}

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, **kwargs)
    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(azure_deployment=model, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, **kwargs)
    if provider == "heuristic":
        return HeuristicLLM()
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def build_embeddings(embedding_config: dict):
    """Build a LangChain embedding model from config.

    Config keys: provider (gemini | azure_openai | hash), model.
    """
    provider = embedding_config["provider"]
    model = embedding_config["model"]

    if provider == "gemini":
        return GeminiEmbeddings(model=model)
    if provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(azure_deployment=model)
    if provider == "hash":
        return HashEmbeddings()
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def load_documents(data_source: str) -> list[Document]:
    """Load .txt, .md, and .pdf files from a directory using LangChain loaders.

    Args:
        data_source: Path to the directory containing source documents.

    Returns:
        Flat list of LangChain Document objects.
    """
    path = Path(data_source)
    docs: list[Document] = []

    for glob_pattern in ("**/*.txt", "**/*.md"):
        for file_path in path.glob(glob_pattern):
            try:
                docs.append(
                    Document(
                        page_content=file_path.read_text(encoding="utf-8"),
                        metadata={"source": str(file_path)},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("text_load_failed", path=str(file_path), error=str(exc))

    try:
        from pypdf import PdfReader

        for pdf_path in path.glob("**/*.pdf"):
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                docs.append(
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={"source": str(pdf_path), "page": page_number},
                    )
                )
    except Exception:
        logger.warning("pypdf_unavailable", detail="PDF files will be skipped")

    logger.info("documents_loaded", count=len(docs), source=str(path))
    return docs


def chunk_documents(
    documents: list[Document],
    chunking_config: dict,
    embeddings=None,
) -> list[Document]:
    """Chunk documents using the strategy declared in config.

    Config keys:
        strategy: fixed | semantic | parent_child
        chunk_size: token target per chunk (default 512)
        overlap: overlap between consecutive chunks (default 50)

    The 'semantic' strategy uses SemanticChunker when embeddings are provided;
    it falls back to RecursiveCharacterTextSplitter otherwise.
    The 'parent_child' strategy stores parent text in child chunk metadata.
    """
    strategy = chunking_config.get("strategy", "fixed")
    chunk_size = chunking_config.get("chunk_size", 512)
    overlap = chunking_config.get("overlap", 50)

    def split_text(text: str, size: int, chunk_overlap: int) -> list[str]:
        if size <= 0:
            return [text]
        step = max(1, size - chunk_overlap)
        return [text[i : i + size] for i in range(0, len(text), step)]

    def split_documents_local(source_documents: list[Document], size: int, chunk_overlap: int) -> list[Document]:
        split_docs: list[Document] = []
        for doc in source_documents:
            for index, chunk in enumerate(split_text(doc.page_content, size, chunk_overlap)):
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk_index": index},
                    )
                )
        return split_docs

    if strategy == "fixed":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            return splitter.split_documents(documents)
        except Exception:
            return split_documents_local(documents, chunk_size, overlap)

    if strategy == "semantic":
        if embeddings is not None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker

                splitter = SemanticChunker(embeddings)
                return splitter.split_documents(documents)
            except Exception:
                logger.warning("semantic_chunker_unavailable", fallback="fixed")
        return split_documents_local(documents, chunk_size, overlap)

    if strategy == "parent_child":
        parent_docs = split_documents_local(documents, chunk_size, overlap)
        child_docs: list[Document] = []
        for parent_id, parent in enumerate(parent_docs):
            for child in split_documents_local([parent], max(1, chunk_size // 2), overlap // 2):
                child.metadata["parent_content"] = parent.page_content
                child.metadata["parent_id"] = parent_id
                child_docs.append(child)
        return child_docs

    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def build_vector_store(
    chunks: list[Document],
    embeddings,
    collection: str,
):
    """Build a Chroma vector store from a list of chunked documents.

    Args:
        chunks: Pre-chunked LangChain Documents.
        embeddings: LangChain embedding model.
        collection: Chroma collection name (one per experiment to avoid cross-contamination).

    Returns:
        Populated Chroma instance ready for similarity search.
    """
    chunks = [c for c in chunks if c.page_content.strip()]
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    try:
        from langchain_chroma import Chroma

        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection,
            persist_directory=persist_dir,
        )
    except Exception:
        logger.warning("chroma_unavailable", fallback="in_memory")
        return InMemoryVectorStore(chunks, embeddings)


def build_retriever(
    retrieval_config: dict,
    vector_store,
    chunks: list[Document],
):
    """Build a dense, sparse, or hybrid retriever from config.

    Config keys:
        strategy: dense | sparse | hybrid
        top_k: number of results to retrieve
        alpha: weight for dense retriever in hybrid (0–1, default 0.5)
    """
    strategy = retrieval_config.get("strategy", "dense")
    top_k = retrieval_config.get("top_k", 5)

    if strategy == "dense":
        return vector_store.as_retriever(search_kwargs={"k": top_k})

    if strategy == "sparse":
        try:
            from langchain_community.retrievers import BM25Retriever

            return BM25Retriever.from_documents(chunks, k=top_k)
        except Exception:
            return SparseRetriever(chunks, k=top_k)

    if strategy == "hybrid":
        alpha = retrieval_config.get("alpha", 0.5)
        dense = vector_store.as_retriever(search_kwargs={"k": top_k})
        try:
            from langchain.retrievers import EnsembleRetriever
            from langchain_community.retrievers import BM25Retriever

            sparse = BM25Retriever.from_documents(chunks, k=top_k)
            return EnsembleRetriever(retrievers=[dense, sparse], weights=[alpha, 1 - alpha])
        except Exception:
            sparse = SparseRetriever(chunks, k=top_k)
            return HybridRetriever(dense, sparse, alpha=alpha, k=top_k)

    raise ValueError(f"Unknown retrieval strategy: {strategy!r}")


def wrap_reranker(retriever, reranker_config: dict):
    """Optionally wrap a retriever with cross-encoder reranking.

    Falls back to the original retriever if cross-encoder dependencies are missing.
    Config keys: enabled, top_k_after_rerank.
    """
    if not reranker_config.get("enabled", False):
        return retriever

    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_n = reranker_config.get("top_k_after_rerank", 5)
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
    except ImportError:
        logger.warning("reranker_unavailable", reason="sentence-transformers not installed")
        return retriever
