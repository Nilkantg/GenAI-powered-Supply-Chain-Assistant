# main.py
import pandas as pd
from data_loaders.text_splitter import load_data, split_text, TextSplitter, LoadedData
from Embedding_layer.embedding import embedding_, DataLoader
from Vector_Store_db.vectorstore import create_vector_store, Vector_loader
from Model_building.groq_model import build_model, Model
from Tools.tools2 import tool_binding
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

    # Correct CSV file path
    file_path = "Datasets/Supply_data_1/supply_chain_forecast.csv"

    # Load and process CSV
    data = LoadedData(file_path=file_path)
    df = load_data(data)

    # Combine all columns into a string and create metadata
    text_content = df.apply(lambda row: ", ".join(row.astype(str)), axis=1).tolist()
    metadatas = [{"sku": row["SKU"], "index": i} for i, (_, row) in enumerate(df.iterrows())]

    # Split text into chunks
    text_params = TextSplitter(text=" ".join(text_content), chunk_size=1000, chunk_overlap=200)
    text_documents = split_text(text_params)

    # Create metadata for each chunk (simplified: assign index-based metadata)
    chunk_metadatas = [{"chunk_id": i, "source": "supply_chain_forecast"} for i in range(len(text_documents))]

    # Create embeddings
    documents_ = DataLoader(HF_TOKEN=HF_TOKEN, text=text_documents)
    embedding_vectors = embedding_(documents_)

    # Create vector store
    vectors_ = Vector_loader(vectors=embedding_vectors, texts=text_documents, metadatas=chunk_metadatas)
    vector_db = create_vector_store(vectors_)

    # Get user query
    query = input("Enter your query: ")
    if not query:
        raise ValueError("Query cannot be empty.")
    user_query = UserQueryInput(query=query)
    query_ = query_user_input(user_query)

    # Bind tools
    agent_tools = tool_binding()

    # Build model input
    model_input = Model(
        query=query_,
        groq_api_key=GROQ_API_KEY,
        vector_store=vector_db,
        tools=agent_tools
    )

    # Run model
    result = build_model(model_input)

    return {"response": result["response"]}

if __name__ == "__main__":
    response = main()
    print(response)