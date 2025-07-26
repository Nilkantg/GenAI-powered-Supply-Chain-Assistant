import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import os

# Set Groq API key (replace with your actual key)
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# 1. Load and Prepare Dataset
df = pd.read_csv('supply_chain_data_with_forecast.csv')

# Convert dataset rows to text documents for vector store
documents = [
    f"SKU: {row['SKU']}, Product type: {row['Product type']}, Stock levels: {row['Stock levels']}, "
    f"Demand Forecast: {row['Demand Forecast']}, Price: {row['Price']}, Availability: {row['Availability']}, "
    f"Number of products sold: {row['Number of products sold']}, Revenue generated: {row['Revenue generated']}, "
    f"Lead times: {row['Lead times']}, Supplier name: {row['Supplier name']}"
    for _, row in df.iterrows()
]

# 2. Set Up Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(documents, embeddings)

# 3. Define Pydantic Schemas for Tools
class DataRetrievalInput(BaseModel):
    sku: str = Field(description="The SKU identifier to retrieve data for")
    fields: list[str] = Field(description="List of fields to retrieve (e.g., ['Stock levels', 'Demand Forecast'])")

class EOQInput(BaseModel):
    sku: str = Field(description="The SKU identifier for EOQ calculation")
    ordering_cost: float = Field(description="Cost per order (default: 50)", default=50.0)
    holding_cost_per_unit: float = Field(description="Holding cost per unit per period (default: 2)", default=2.0)

class ReorderPointInput(BaseModel):
    sku: str = Field(description="The SKU identifier for reorder point calculation")
    safety_stock: int = Field(description="Safety stock level (default: 50)", default=50)

class ReportInput(BaseModel):
    sku: str = Field(description="The SKU identifier for the report")
    fields: list[str] = Field(description="List of fields to include in the report")

# 4. Define Custom Tools
def retrieve_data(sku: str, fields: list[str]) -> str:
    """Retrieve specified fields for a given SKU from the dataset."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    result = {field: row[field].iloc[0] for field in fields if field in df.columns}
    return str(result)

def calculate_eoq(sku: str, ordering_cost: float = 50.0, holding_cost_per_unit: float = 2.0) -> str:
    """Calculate Economic Order Quantity (EOQ) for a given SKU."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    demand = row['Demand Forecast'].iloc[0]
    eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost_per_unit)
    return f"EOQ for SKU {sku}: {round(eoq, 2)} units"

def calculate_reorder_point(sku: str, safety_stock: int = 50) -> str:
    """Calculate reorder point for a given SKU."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    demand = row['Demand Forecast'].iloc[0]
    lead_time = row['Lead times'].iloc[0]
    daily_demand = demand / 30  # Assume monthly demand
    reorder_point = (daily_demand * lead_time) + safety_stock
    return f"Reorder point for SKU {sku}: {round(reorder_point, 2)} units"

def generate_report(sku: str, fields: list[str]) -> str:
    """Generate a Markdown report for the specified SKU and fields."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    report = f"# Inventory Report for SKU {sku}\n\n"
    for field in fields:
        if field in df.columns:
            report += f"**{field}**: {row[field].iloc[0]}\n"
    # Calculate EOQ and Reorder Point for the report
    demand = row['Demand Forecast'].iloc[0]
    lead_time = row['Lead times'].iloc[0]
    eoq = np.sqrt((2 * demand * 50.0) / 2.0)  # Default values
    daily_demand = demand / 30
    reorder_point = (daily_demand * lead_time) + 50
    report += f"\n**EOQ**: {round(eoq, 2)} units\n"
    report += f"**Reorder Point**: {round(reorder_point, 2)} units\n"
    # Save report to file
    with open(f'report_{sku}.md', 'w') as f:
        f.write(report)
    return f"Report generated for SKU {sku} and saved as report_{sku}.md"

# Create LangChain tools
tools = [
    StructuredTool.from_function(
        func=retrieve_data,
        name="RetrieveData",
        description="Retrieve specific fields for a given SKU from the dataset.",
        args_schema=DataRetrievalInput
    ),
    StructuredTool.from_function(
        func=calculate_eoq,
        name="CalculateEOQ",
        description="Calculate Economic Order Quantity (EOQ) for a given SKU using demand, ordering cost, and holding cost.",
        args_schema=EOQInput
    ),
    StructuredTool.from_function(
        func=calculate_reorder_point,
        name="CalculateReorderPoint",
        description="Calculate reorder point for a given SKU using demand, lead time, and safety stock.",
        args_schema=ReorderPointInput
    ),
    StructuredTool.from_function(
        func=generate_report,
        name="GenerateReport",
        description="Generate a Markdown report for a given SKU with specified fields, including EOQ and reorder point.",
        args_schema=ReportInput
    )
]

# 5. Set Up Agent
llm = ChatGroq(model="grok-3", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an inventory management assistant. Use the dataset and tools to answer queries about stock levels, demand, EOQ, reorder points, or generate reports. For data queries, use the RetrieveData tool. For calculations, use CalculateEOQ or CalculateReorderPoint. For reports, use GenerateReport. If the query is ambiguous, search the vector store for context: {vector_store_search}."),
    ("human", "{input}"),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{agent_scratchpad}")
])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 6. Example Queries
queries = [
    "What are the stock levels and demand for SKU04?",
    "Calculate EOQ for SKU04",
    "Generate a report for SKU04 with fields Stock levels, Demand Forecast, Price"
]

for query in queries:
    print(f"\nQuery: {query}")
    result = agent_executor.invoke({"input": query, "vector_store_search": vector_store.search(query, k=1)[0][0]})
    print(f"Response: {result['output']}")