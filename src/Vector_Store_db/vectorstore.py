# Vector_Store_db/vectorstore.py
import chromadb
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field, validator
from typing import List

class Vector_loader(BaseModel):
    """Model to hold embedded vectors and their corresponding texts."""
    vectors: List[List[float]] = Field(..., description="List of embedded vectors from embedding file")
    texts: List[str] = Field(..., description="List of original text documents")
    metadatas: List[dict] = Field(default_factory=list, description="List of metadata dictionaries for each document")

    @validator("metadatas")
    def check_non_empty_metadata(cls, v, values):
        """Ensure metadata dictionaries are non-empty."""
        if not v:  # If metadatas is empty, create default non-empty metadata
            return [{"chunk_id": i} for i in range(len(values.get("texts", [])))]
        for i, meta in enumerate(v):
            if not meta:  # Replace empty dict with default
                v[i] = {"chunk_id": i}
        return v

def create_vector_store(vectors_: Vector_loader, persist_directory: str = "chroma_db") -> Chroma:
    """Create a Chroma vector store from precomputed embeddings, texts, and metadata."""
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get a collection
    collection_name = "supply_chain_vectors"
    collection = client.get_or_create_collection(name=collection_name)
    
    # Prepare data for Chroma
    ids = [f"doc_{i}" for i in range(len(vectors_.texts))]
    embeddings = vectors_.vectors
    texts = vectors_.texts
    metadatas = vectors_.metadatas
    
    # Validate lengths
    if len(embeddings) != len(texts) or len(texts) != len(metadatas):
        raise ValueError("Length mismatch: embeddings, texts, and metadatas must have the same length")
    
    # Add embeddings, texts, and metadata to the collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    # Create LangChain Chroma wrapper for compatibility
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors_db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    
    return vectors_db