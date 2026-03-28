"""Build LangChain components from YAML experiment configs."""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)


def build_llm(llm_config: dict):
    """Build a LangChain chat model from a provider/model config dict.

    Config keys: provider (azure_openai | anthropic), model, temperature, max_tokens.
    """
    provider = llm_config["provider"]
    model = llm_config["model"]
    kwargs = {k: v for k, v in llm_config.items() if k not in {"provider", "model"}}

    if provider == "azure_openai":
        return AzureChatOpenAI(azure_deployment=model, **kwargs)
    if provider == "anthropic":
        return ChatAnthropic(model=model, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def build_embeddings(embedding_config: dict) -> AzureOpenAIEmbeddings:
    """Build a LangChain embedding model from config.

    Config keys: provider (azure_openai), model.
    """
    provider = embedding_config["provider"]
    model = embedding_config["model"]

    if provider == "azure_openai":
        return AzureOpenAIEmbeddings(azure_deployment=model)
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
        loader = DirectoryLoader(
            str(path),
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
            show_progress=False,
        )
        docs.extend(loader.load())

    try:
        from langchain_community.document_loaders import PyPDFLoader

        for pdf_path in path.glob("**/*.pdf"):
            docs.extend(PyPDFLoader(str(pdf_path)).load())
    except ImportError:
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

    if strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_documents(documents)

    if strategy == "semantic":
        if embeddings is not None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker

                splitter = SemanticChunker(embeddings)
                return splitter.split_documents(documents)
            except ImportError:
                logger.warning("semantic_chunker_unavailable", fallback="fixed")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_documents(documents)

    if strategy == "parent_child":
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size // 2, chunk_overlap=overlap // 2
        )
        parent_docs = parent_splitter.split_documents(documents)
        child_docs: list[Document] = []
        for parent_id, parent in enumerate(parent_docs):
            for child in child_splitter.split_documents([parent]):
                child.metadata["parent_content"] = parent.page_content
                child.metadata["parent_id"] = parent_id
                child_docs.append(child)
        return child_docs

    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def build_vector_store(
    chunks: list[Document],
    embeddings,
    collection: str,
) -> Chroma:
    """Build a Chroma vector store from a list of chunked documents.

    Args:
        chunks: Pre-chunked LangChain Documents.
        embeddings: LangChain embedding model.
        collection: Chroma collection name (one per experiment to avoid cross-contamination).

    Returns:
        Populated Chroma instance ready for similarity search.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )


def build_retriever(
    retrieval_config: dict,
    vector_store: Chroma,
    chunks: list[Document],
):
    """Build a dense, sparse, or hybrid retriever from config.

    Config keys:
        strategy: dense | sparse | hybrid
        top_k: number of results to retrieve
        alpha: weight for dense retriever in hybrid (0–1, default 0.5)
    """
    from langchain.retrievers import EnsembleRetriever

    strategy = retrieval_config.get("strategy", "dense")
    top_k = retrieval_config.get("top_k", 5)

    if strategy == "dense":
        return vector_store.as_retriever(search_kwargs={"k": top_k})

    if strategy == "sparse":
        return BM25Retriever.from_documents(chunks, k=top_k)

    if strategy == "hybrid":
        alpha = retrieval_config.get("alpha", 0.5)
        dense = vector_store.as_retriever(search_kwargs={"k": top_k})
        sparse = BM25Retriever.from_documents(chunks, k=top_k)
        return EnsembleRetriever(retrievers=[dense, sparse], weights=[alpha, 1 - alpha])

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
