import pandas as pd
from data_loaders.text_splitter import load_data, text_splitter, LoadedData, TextSplitter
from Embedding_layer.embedding import embedding_, DataLoader
from Vector_Store_db.vectorstore import create_vector_store, Vector_loader
from Model_building.groq_model import build_model
from Model_building.tools import tool_binding
from Model_building.report_generator import generate_rag_report
from user_query import query_user_input, UserQueryInput
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
import os
from dotenv import load_dotenv
load_dotenv()

def main():

    """Main function to generate the response."""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    file_path = pd.read_csv("Datasets/Supply_data_1/supply_chain_forecast.csv")

    data = LoadedData(file_path=file_path)

    data = load_data(data)

    text_content = " ".join(data.apply(lambda row: " ".join(row.astype(str)), axis=1))

    text_params = TextSplitter(text=text_content)

    text_documents = text_splitter(text_params)

    documents_ = DataLoader(HF_TOKEN=HF_TOKEN, text=text_documents)

    embedding_vectors = embedding_(documents_)

    vectors_ = Vector_loader(vectors=embedding_vectors)

    vector_db = create_vector_store(vectors_)

    query = input("Enter your query: ")
    if not query:
        raise ValueError("Query cannot be empty.")
    user_query = UserQueryInput(query=query)

    query_ = query_user_input(user_query)

    agent_tools = tool_binding()

    result = build_model(query_, GROQ_API_KEY, vector_db, agent_tools)

    return {"response": result["output"]}

if __name__ == "__main__":
    response = main()
    print(response)