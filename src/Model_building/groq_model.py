# groq_model.py
import pandas as pd
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from Tools.tools2 import tool_binding
import os
from dotenv import load_dotenv

load_dotenv()

class Model(BaseModel):
    query: str = Field(..., description="The query to be processed by the model.")
    groq_api_key: Annotated[str, Field(description="Groq API key for authentication.")]
    vector_store: Optional[object] = Field(None, description="Chroma vector store for dataset retrieval")
    tools: List[object] = Field(..., description="List of tools to be used by the agent") # tools should not have a default_factory here
    # It must be passed from main.py after the vector store is created.

def build_model(model: Model) -> dict:
    """
    Build and run an agent to process inventory-related queries using the dataset and vector store.
    
    Args:
        model: Model object containing the user query, Groq API key, vector store, and tools.
    
    Returns:
        Dict with 'response' key containing the agent's output and intermediate steps.
    """
    llm = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=model.groq_api_key,
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        output_key="output"
    )

    vector_store_search = ""
    if model.vector_store is not None:
        docs = model.vector_store.similarity_search(model.query, k=3)
        vector_store_search = "\n".join([f"SKU: {doc.metadata['sku']}, Data: {doc.page_content}" for doc in docs])

    # --- CORRECTED PROMPT ---
    # The prompt now explicitly lists the new tool and provides instructions for when to use it.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an inventory management assistant. Your task is to process queries about stock levels, demand, EOQ, reorder points, or generate reports. Follow these steps:
        1. Parse the query to identify the SKU (e.g., 'SKU1') and the task (retrieve data, calculate EOQ, calculate reorder point, or generate report).
        2. Extract parameters from the query (e.g., fields for RetrieveData, ordering_cost for CalculateEOQ).
        3. Use the appropriate tool based on the task:
        - Use RetrieveData for queries about stock levels, demand, or other fields (e.g., 'get stock for SKU4').
        - Use CalculateEOQ for EOQ calculations (e.g., 'EOQ for SKU1').
        - Use CalculateReorderPoint for reorder point calculations (e.g., 'reorder point for SKU1').
        - **Use GenerateReport for comprehensive reports (e.g., 'report for SKU0').**
        4. If the query is ambiguous, use the vector store context to find relevant SKUs: {vector_store_search}.
        5. Validate inputs using the tool's schema (e.g., ensure SKU exists, fields are valid).
        6. If no tool is applicable, respond with a summary based on the vector store context.
        Return concise, accurate responses. If parameters are missing, use defaults from the tool schemas."""),
        ("human", "{query}"),
        ("placeholder", "{chat_history}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, model.tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=model.tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    result = agent_executor.invoke({
        "query": model.query,
        "vector_store_search": vector_store_search
    })

    return result