from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import os

class ReportGeneratorInput(BaseModel):
    sku: str = Field(description="The SKU identifier for the report")
    fields: list[str] = Field(description="List of fields to include in the report")
    groq_api_key: str = Field(description="Groq API key for authentication")
    vector_store: object = Field(description="FAISS vector store for dataset retrieval")

def generate_rag_report(sku: str, fields: list[str], groq_api_key: str, vector_store: object) -> str:
    """Generate a Markdown report for the specified SKU using RAG."""
    # Initialize LLM
    llm = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=groq_api_key,
        temperature=0.3
    )

    # Initialize memory for RAG chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Define prompt template for report generation
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an inventory management assistant. Using the provided dataset context, generate a Markdown report for SKU {sku} with the specified fields: {fields}.
        Include Economic Order Quantity (EOQ) and reorder point calculations.
        - Use demand from 'Demand Forecast' and assume monthly demand.
        - EOQ = sqrt((2 * demand * ordering_cost) / holding_cost_per_unit), with ordering_cost=50, holding_cost_per_unit=2.
        - Reorder point = (daily_demand * lead_time) + safety_stock, with daily_demand=demand/30, safety_stock=50.
        Format the report in Markdown with clear headings and bullet points.
        Context: {context}
        Question: Generate a report for SKU {sku} with fields {fields}.
        """
    )

    # Create RAG chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    # Run RAG chain
    query = f"Generate a report for SKU {sku} with fields {', '.join(fields)}."
    result = rag_chain({"question": query})

    # Save report to file
    report = result["answer"]
    with open(f'report_{sku}.md', 'w') as f:
        f.write(report)
    
    return f"Report generated for SKU {sku} and saved as report_{sku}.md"