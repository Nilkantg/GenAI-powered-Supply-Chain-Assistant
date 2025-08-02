# report_generator.py
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import os
from typing import List

class ReportGeneratorInput(BaseModel):
    sku: str = Field(description="The SKU identifier for the report")
    fields: List[str] = Field(description="List of fields to include in the report")
    groq_api_key: str = Field(description="Groq API key for authentication")
    vector_store: object = Field(description="FAISS vector store for dataset retrieval")

def generate_rag_report(sku: str, fields: List[str], groq_api_key: str, vector_store: object) -> str:
    """Generate a Markdown report for the specified SKU using RAG."""
    llm = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=groq_api_key,
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        You are an inventory management assistant. Using the provided context for SKU {sku}, generate a comprehensive Markdown report.
        
        The report should include all data fields present in the context.
        Additionally, perform the following calculations and include them in the report:
        
        - **Economic Order Quantity (EOQ):** Calculate EOQ using demand from 'Demand Forecast', with an ordering cost of $50 and a holding cost per unit of $2. Use the formula: EOQ = sqrt((2 * demand * 50) / 2).
        - **Reorder Point:** Calculate the reorder point using 'Demand Forecast' and 'Lead times'. Assume daily demand is demand/30 and safety stock is 50. Use the formula: Reorder Point = (daily_demand * lead_time) + 50.

        Format the report in Markdown with clear headings and bullet points for all data and calculations.
        
        Context: {{context}}
        Question: {{question}}
        """
    )
    
    # --- CORRECTION: Add metadata filter to the retriever ---
    # The `as_retriever` method accepts a `search_kwargs` dictionary. We can use the 'filter' key
    # to specify that we only want documents where the 'sku' metadata field matches the
    # SKU from the user's query.
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "sku": sku  # Filter documents by the specific SKU
            }
        }
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, # Use the configured retriever
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    # The query is a direct instruction for the RAG chain.
    query = f"Generate a detailed inventory report for SKU {sku} including all available data and calculated metrics."
    
    result = rag_chain.invoke({
        "question": query
    })

    report = result["answer"]
    
    report_file_path = f'report_{sku}.md'
    with open(report_file_path, 'w') as f:
        f.write(report)
    
    return f"Report generated for SKU {sku} and saved as {report_file_path}"