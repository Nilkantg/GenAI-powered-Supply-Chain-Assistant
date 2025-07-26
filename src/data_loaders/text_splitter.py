from langchain.document_loaders import PyPDFLoader #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
import pandas as pd 
import pickle
import os
import pydantic
from typing import List, Dict, Literal, Any, Annotated, Optional
from pydantic import BaseModel, Field 

class loaded_data(BaseModel):
    """Model to represent the loaded data."""
    file_path: str = Field(..., description="Path to the data file.")
    # data: pd.DataFrame = Field(..., description="The loaded DataFrame containing supply chain data.")

def load_data(data: loaded_data) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(loaded_data.file_path)

class text_splitter(BaseModel):
    """Model to represent text splitting parameters."""
    chunk_size: int = Field(default=1000, description="Size of each text chunk.")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks.")
    text: Annotated[str, Field(description="Text to be split into chunks.")]

def text_splitter(text: text_splitter) -> List[str]:
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text.chunk_size, 
        chunk_overlap=text.chunk_overlap
    )
    splitted_text = text_splitter.split_text(text)

    return splitted_text



