import pandas as pd
from data_loaders.text_splitter import load_data, text_splitter
from Embedding_layer.embedding import embedding_
from Vector_Store_db.vectorstore import create_vector_store
from Model_building.groq_model import build_model
from Model_building.tools import tool_binding
from Model_building.report_generator import generate_rag_report
from user_query import query_user_input
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
import os
from dotenv import load_dotenv
load_dotenv()

def main():

    """Main function to generate the response."""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    path = pd.read_csv("Datasets/Supply_data_1/supply_chain_forecast.csv")

    data = load_data(path)

    text_documents = text_splitter(data)

    embedding_vectors = embedding_(text_documents)

    vector_db = create_vector_store(embedding_vectors)

    query_ = query_user_input()

    agent_tools = tool_binding()

    result = build_model(query_, GROQ_API_KEY, vector_db, agent_tools)

    return {"response": result["output"]}

if __name__ == "__main__":
    response = main()
    print(response)