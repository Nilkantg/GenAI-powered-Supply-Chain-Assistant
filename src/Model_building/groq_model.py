import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
import os
from dotenv import load_dotenv
load_dotenv()

# Pydantic schema for query input
class Model(BaseModel):
    query: str = Field(..., description="The query to be processed by the model.")
    groq_api_key: Annotated[str, Field(description="Groq API key for authentication.")]
    vector_store: Optional[object] = Field(None, description="FAISS vector store for dataset retrieval")
    tools: List[StructuredTool] = Field(default_factory=list, description="List of tools to be used by the agent")


def build_model(model: Model) -> Dict[str, str]:
    """
    Build and run an agent to process inventory-related queries using the dataset and vector store.
    
    Args:
        query: Query object containing the user query, Groq API key, and vector store.
    
    Returns:
        Dict with 'response' key containing the agent's output.
    """
    # Initialize LLM
    llm = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=model.groq_api_key,
        temperature=0.3
    )

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        output_key="response"
    )
    # # Set the path relative to your Model_building folder
    # persist_directory = "../Vector_Store_db/chroma_db"

    # # Load the existing Chroma vector store
    # vector_store = Chroma(persist_directory=persist_directory)

    # Retrieve context from vector store if available
    vector_store_search = ""
    if model.vector_store is not None:
        # Most vector stores have a 'similarity_search' method
        docs = model.vector_store.similarity_search(model.query, k=3)
        # Combine retrieved docs into a single string
        vector_store_search = "\n".join([doc.page_content for doc in docs])

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an inventory management assistant. Use the dataset and tools to answer queries about stock levels, demand, EOQ, reorder points, or generate reports. For data queries, use RetrieveData. For calculations, use CalculateEOQ or CalculateReorderPoint. For reports, use GenerateReport. If the query is ambiguous, search the vector store for context: {vector_store_search}."),
        ("human", "{query}"),
        ("placeholder", "{chat_history}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Create agent
    agent = create_tool_calling_agent(llm, model.tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=model.tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    # Run the agent, passing the retrieved context
    result = agent_executor.invoke({
        "query": model.query,
        "vector_store_search": vector_store_search
    })

    # return {"response": result["output"]}
    return result

