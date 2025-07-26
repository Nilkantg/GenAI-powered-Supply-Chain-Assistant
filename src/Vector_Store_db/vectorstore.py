from langchain.vectorstores import FAISS, Chroma
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
import os
import pandas as pd
import pickle
from dotenv import load_dotenv #type: ignore
load_dotenv()

class Vector_loader(BaseModel):
    """It takes the vectors from embedding file"""
    vectors: Annotated[int, Field(description="The vector data from embedding file")]

def vector_store(vectors: Vector_loader, persist_directory: str = "chroma_db"):
    # Create or load a Chroma vector store with a persistence directory
    vectors_db = Chroma.from_documents(
        vectors.vectors,  # assuming vectors.vectors is a list of documents
        persist_directory=persist_directory
    )
    # Save the vector store to disk
    vectors_db.persist()
    
    return vectors_db