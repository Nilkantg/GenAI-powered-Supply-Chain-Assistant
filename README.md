# GenAI-powered-Supply-Chain-Assistant
<!-- Built a GenAI-powered supply chain assistant that integrates LLM-based conversational queries, demand forecasting, supplier risk evaluation, and automated document generation. Implemented using LangChain, ML models, and Streamlit. -->

📜 Project Description
The GenAI-Powered Supply Chain Assistant is an intelligent system designed to help supply chain managers make data-driven decisions. By integrating a Large Language Model (LLM) with a Retrieval-Augmented Generation (RAG) architecture, this assistant can answer complex queries about inventory, stock levels, and demand forecasts. It also provides tools for calculating key metrics like Economic Order Quantity (EOQ) and Reorder Points, and can even generate comprehensive inventory reports on demand.

The project is built around a modular pipeline that:

Loads and processes supply chain data.

Creates a vector store for efficient semantic search.

Utilizes a conversational agent to interpret user queries.

Invokes specialized tools and RAG chains to retrieve information and perform calculations.

✨ Features
Intelligent Query Answering: Ask natural language questions about your inventory, stock levels, and product-specific data.

Tool-Based Calculations: The agent can automatically use specialized tools to calculate:

Economic Order Quantity (EOQ)

Reorder Points

Automated Report Generation: Generate detailed, well-formatted Markdown reports for any SKU, including data from the database and calculated metrics.

Persistent Vector Store: The system uses a ChromaDB vector store that is saved to disk, allowing for faster loading times on subsequent runs.

Modular and Extensible: The project's structure is organized into separate components (data loaders, embeddings, tools, models), making it easy to add new features or integrate different models.

🛠️ Technology Stack
Language Model: Groq (Gemma2-9b-It)

Framework: LangChain

Vector Store: ChromaDB

Embeddings: HuggingFace sentence-transformers/all-MiniLM-L6-v2

Data Handling: Pandas

Environment Management: python-dotenv

🚀 Getting Started
Prerequisites
Python 3.9 or later
A Groq API Key
A Hugging Face Hub API Token (for embeddings)
