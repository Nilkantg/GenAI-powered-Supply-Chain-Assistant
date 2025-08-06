import pandas as pd
import numpy as np
import os

# Read the CSV file
# Replace 'supply_chain_data.csv' with the path to your dataset
df = pd.read_csv('D:\Gen_AI\Supply Chain\Datasets\Supply_data_1\supply_chain_data.csv')

# Function to calculate adjustment factor for demand forecast
def calculate_adjustment_factor(row):
    sales = row['Number of products sold']
    stock = row['Stock levels']
    availability = row['Availability']
    lead_time = row['Lead times']
    
    adjustment = 0
    
    # Stock level adjustment
    if stock > 2 * sales:
        adjustment += 0.10  # +10% for high stock
    elif stock < 0.5 * sales:
        adjustment -= 0.05  # -5% for low stock
    
    # Availability adjustment
    if availability > 50:
        adjustment += 0.05  # +5% for high availability
    elif availability < 20:
        adjustment -= 0.05  # -5% for low availability
    
    # Lead time adjustment
    if lead_time < 10:
        adjustment += 0.05  # +5% for short lead time
    elif lead_time > 20:
        adjustment -= 0.05  # -5% for long lead time
    
    # Cap adjustment factor between -15% and +20%
    adjustment = max(min(adjustment, 0.20), -0.15)
    
    return adjustment

# Calculate demand forecast
df['Demand Forecast'] = df.apply(
    lambda row: round(row['Number of products sold'] * (1 + calculate_adjustment_factor(row))),
    axis=1
)

# Reorder columns to place 'Demand Forecast' after 'Costs'
columns = df.columns.tolist()
costs_index = columns.index('Costs')
new_columns = columns[:costs_index + 1] + ['Demand Forecast'] + columns[costs_index + 1:-1]
df = df[new_columns]

# Save the updated dataset to a new CSV file
# df.to_csv('supply_chain_data_with_forecast.csv', index=False)

# Display the first few rows to verify
print(df[['SKU', 'Product type', 'Number of products sold', 'Demand Forecast']].head(10))

# Optional: Save a subset of columns for quick reference
df_subset = df.drop(['Routes', 'Customer demographics', 'Supplier name'], axis=1)

# Ensure the directory exists (absolute path)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'Datasets', 'Supply_data_1')
os.makedirs(output_dir, exist_ok=True)

# Save the subset to CSV
output_path = os.path.join(output_dir, 'supply_chain_forecast.csv')
print(f"Saving forecast data to {output_path}")
df_subset.to_csv(output_path, index=False)
print(f"Forecast data saved to {output_path}")
