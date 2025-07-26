import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
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

    # Perform vector store search for context (if vector_store is provided)
    vector_store_search = ""
    if model.vector_store:
        search_results = model.vector_store.search(model.query, k=1)
        vector_store_search = search_results[0][0] if search_results else ""

    # Run the agent
    result = agent_executor.invoke({
        "query": model.query,
        "vector_store_search": vector_store_search
    })

    return {"response": result["output"]}

