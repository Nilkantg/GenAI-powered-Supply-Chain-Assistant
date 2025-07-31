from langchain.document_loaders import PyPDFLoader #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
import pandas as pd 
import pickle
import os
import pydantic
from typing import List, Dict, Literal, Any, Annotated, Optional
from pydantic import BaseModel, Field 
import logging

# Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class LoadedData(BaseModel):
    """Model to represent the loaded data."""
    file_path: str = Field(..., description="Path to the data file.")

def load_data(data: LoadedData) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(data.file_path)

class TextSplitter(BaseModel):
    """Model to represent text splitting parameters."""
    chunk_size: int = Field(default=1000, description="Size of each text chunk.")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks.")
    text: Annotated[str, Field(description="Text to be split into chunks.")]

def split_text(params: TextSplitter) -> List[str]:
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=params.chunk_size, 
        chunk_overlap=params.chunk_overlap
    )
    return text_splitter.split_text(params.text)

# if __name__ == "__main__":
#     # Example usage
#     file_path = r'Datasets\Supply_data_1\supply_chain_forecast.csv'
    
#     # Create a LoadedData instance
#     data = LoadedData(file_path=file_path)
#     df = load_data(data)
#     print(df.head())

#     # Combine all columns into a single string
#     text_content = " ".join(df.apply(lambda row: " ".join(row.astype(str)), axis=1))
#     text_params = TextSplitter(text=text_content)
#     chunks = split_text(text_params)
#     print(chunks)

