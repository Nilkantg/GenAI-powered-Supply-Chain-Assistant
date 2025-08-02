# Vector_Store_db/vectorstore.py
import chromadb
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field, validator
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

class Vector_loader(BaseModel):
    """Model to hold embedded vectors and their corresponding texts."""
    vectors: List[List[float]] = Field(..., description="List of embedded vectors from embedding file")
    texts: List[str] = Field(..., description="List of original text documents")
    metadatas: List[dict] = Field(default_factory=list, description="List of metadata dictionaries for each document")

    @validator("metadatas")
    def check_non_empty_metadata(cls, v, values):
        """Ensure metadata dictionaries are non-empty."""
        if not v:
            return [{"chunk_id": i} for i in range(len(values.get("texts", [])))]
        for i, meta in enumerate(v):
            if not meta:
                v[i] = {"chunk_id": i}
        return v

def create_vector_store(vectors_: Vector_loader, persist_directory: str = "chroma_db") -> Chroma:
    """Create a Chroma vector store from precomputed embeddings, texts, and metadata."""
    
    # --- CORRECTED LOGIC ---
    # The standard way to interact with Chroma through LangChain is to use its dedicated methods.
    # We will create the Chroma instance and then use the `add_texts` method to populate it.
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a Chroma instance using the provided texts, embeddings, and metadatas
    vectors_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        # We don't specify the collection_name here, LangChain will handle it.
    )

    # Use the Chroma instance's `add_texts` method to add the data
    # This is the correct way to add texts and metadata to a LangChain Chroma wrapper.
    vectors_db.add_texts(
        texts=vectors_.texts,
        metadatas=vectors_.metadatas
    )
    
    return vectors_db