from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
from dotenv import load_dotenv
import os
load_dotenv()

class UserQueryInput(BaseModel):
    GROQ_API_KEY: Optional[Annotated[str, Field(..., description="Groq API key for authentication")]]
    HF_TOKEN: Optional[Annotated[str, Field(..., description="Hugging Face token for model access")]]
    query: Annotated[str, Field(..., description="Enter your query: ")]

def query_user_input(Query: UserQueryInput) -> str:
    """Process user input for generating a RAG report."""

    if not Query.GROQ_API_KEY:
        Query.GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not Query.HF_TOKEN:
        Query.HF_TOKEN = os.getenv("HF_TOKEN")

    return Query.query, Query.GROQ_API_KEY, Query.HF_TOKEN

    