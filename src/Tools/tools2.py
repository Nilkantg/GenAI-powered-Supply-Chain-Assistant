# tools2.py
import pandas as pd
import numpy as np
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional
from Model_building.report_generator import generate_rag_report
from functools import partial # Added functools
import os
from dotenv import load_dotenv

load_dotenv()

# Load dataset
df = pd.read_csv('Datasets/Supply_data_1/supply_chain_forecast.csv')

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

class ReportInput(BaseModel):
    sku: str = Field(description="The SKU identifier for the report")
    fields: Optional[List[str]] = Field(default=None, description="List of fields to include in the report")

def retrieve_data(sku: str, fields: List[str]) -> str:
    """Retrieve specified fields for a given SKU from the dataset."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    
    result = {}
    for field in fields:
        if field in df.columns:
            result[field] = row[field].iloc[0]
        else:
            result[field] = f"Field '{field}' not found in dataset"
    return str(result)

def calculate_eoq(sku: str, ordering_cost: float = 50.0, holding_cost_per_unit: float = 2.0) -> str:
    """Calculate Economic Order Quantity (EOQ) for a given SKU."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    
    if 'Demand Forecast' not in df.columns:
        return "Error: 'Demand Forecast' column not found in dataset"
    
    demand = row['Demand Forecast'].iloc[0]
    if pd.isna(demand):
        return f"Error: Demand Forecast is missing for SKU {sku}"
    
    eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost_per_unit)
    return f"EOQ for SKU {sku}: {round(eoq, 2)} units"

def calculate_reorder_point(sku: str, safety_stock: int = 50) -> str:
    """Calculate reorder point for a given SKU."""
    row = df[df['SKU'] == sku]
    if row.empty:
        return f"No data found for SKU {sku}"
    
    if 'Demand Forecast' not in df.columns or 'Lead times' not in df.columns:
        return "Error: Required columns ('Demand Forecast', 'Lead times') not found in dataset"
    
    demand = row['Demand Forecast'].iloc[0]
    lead_time = row['Lead times'].iloc[0]
    if pd.isna(demand) or pd.isna(lead_time):
        return f"Error: Missing data for SKU {sku}"
    
    daily_demand = demand / 30  # Assume monthly demand
    reorder_point = (daily_demand * lead_time) + safety_stock
    return f"Reorder point for SKU {sku}: {round(reorder_point, 2)} units"

def generate_report_tool(sku: str, fields: Optional[List[str]] = None, groq_api_key: str = None, vector_store: object = None) -> str:
    """Generate a comprehensive report for the specified SKU using RAG."""
    # This tool acts as a bridge to your report_generator function
    if not groq_api_key or not vector_store:
        return "Error: Groq API key or vector store not provided to the report tool."

    if fields is None:
        fields = ["Stock levels", "Demand Forecast", "Lead times", "Order Quantities", "Costs", "Transportation modes"]

    return generate_rag_report(sku, fields, groq_api_key, vector_store)

def tool_binding(groq_api_key: str, vector_store: object):
    """Bind tools for the agent, including the new report generator tool."""
    report_generator_with_deps = partial(
        generate_report_tool,
        groq_api_key=groq_api_key,
        vector_store=vector_store
    )

    return [
        StructuredTool.from_function(
            func=retrieve_data,
            name="RetrieveData",
            description="Retrieve specific fields (e.g., Stock levels, Demand Forecast) for a given SKU from the dataset.",
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
            func=report_generator_with_deps,
            name="GenerateReport",
            description="Generate a comprehensive markdown report for a given SKU. Takes an SKU and an optional list of fields.",
            args_schema=ReportInput
        )
    ]