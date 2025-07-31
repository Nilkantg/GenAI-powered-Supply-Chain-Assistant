from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
import os
import pandas as pd
import pickle
from dotenv import load_dotenv #type: ignore
load_dotenv()

class DataLoader(BaseModel):
    """loads the splitted text from text_splitter"""
    text: Annotated[list[str], Field(description="The data that should be converted from text to embed vectors.")]
    HF_TOKEN: Optional[Annotated[str, Field(description="Provide the HuggingFace API")]]

def embedding_(data: DataLoader) -> List[List[float]]:

    if not data.HF_TOKEN:
        data.HF_TOKEN = os.getenv("HF_TOKEN")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )   
    #huggingfacehub_api_token=data.HF_TOKEN
    embedded_vectors = embedding.embed_documents(data.text)

    return embedded_vectors

