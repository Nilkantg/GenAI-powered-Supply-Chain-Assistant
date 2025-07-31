import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
# from report_generator import generate_rag_report
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load dataset (assumed to be in the same directory)
df = pd.read_csv('supply_chain_data_with_forecast.csv')

# Custom Tools
class DataRetrievalInput(BaseModel):
    sku: str = Field(description="The SKU identifier to retrieve data for")
    fields: List[str] = Field(description="List of fields to retrieve (e.g., ['Stock levels', 'Demand Forecast'])")

class EOQInput(BaseModel):
    sku: str = Field(description="The SKU identifier for EOQ calculation")
    ordering_cost: float = Field(description="Cost per order (default: 50)", default=50.0)
    holding_cost_per_unit: float = Field(description="Holding cost per unit per period (default: 2)", default=2.0)

class ReorderPointInput(BaseModel):
    sku: str = Field(description="The SKU identifier for reorder point calculation")
    safety_stock: int = Field(description="Safety stock level (default: 50)", default=50)

def retrieve_data(sku: str, fields: List[str]) -> str:
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

def tool_binding():
    # Define tools
    Tools = [
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
        )
    ]

    return Tools

# StructuredTool.from_function(
#             func=generate_rag_report,
#             name="GenerateReport",
#             description="Generate a report based on the retrieved data and calculations."
#         )