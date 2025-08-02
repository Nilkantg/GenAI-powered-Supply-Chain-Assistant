# GenAI-powered-Supply-Chain-Assistant
<!-- Built a GenAI-powered supply chain assistant that integrates LLM-based conversational queries, demand forecasting, supplier risk evaluation, and automated document generation. Implemented using LangChain, ML models, and Streamlit. -->

# 📜 Project Description
The GenAI-Powered Supply Chain Assistant is an intelligent system designed to help supply chain managers make data-driven decisions. By integrating a Large Language Model (LLM) with a Retrieval-Augmented Generation (RAG) architecture, this assistant can answer complex queries about inventory, stock levels, and demand forecasts. It also provides tools for calculating key metrics like Economic Order Quantity (EOQ) and Reorder Points, and can even generate comprehensive inventory reports on demand.

The project is built around a modular pipeline that:
Loads and processes supply chain data.
Creates a vector store for efficient semantic search.
Utilizes a conversational agent to interpret user queries.
Invokes specialized tools and RAG chains to retrieve information and perform calculations.

# ✨ Features
- **Intelligent Query Answering:** Ask natural language questions about your inventory, stock levels, and product-specific data.
- **Tool-Based Calculations:** The agent can automatically use specialized tools to  calculate:
    Economic Order Quantity (EOQ)
    Reorder Points
- **Automated Report Generation:** Generate detailed, well-formatted Markdown reports for any SKU, including data from the database and calculated metrics.
- **Persistent Vector Store:** The system uses a ChromaDB vector store that is saved to disk, allowing for faster loading times on subsequent runs.
- **Modular and Extensible:** The project's structure is organized into separate components (data loaders, embeddings, tools, models), making it easy to add new features or integrate different models.

# 🛠️ Technology Stack
- **Language Model:** Groq (Gemma2-9b-It)
- **Framework:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Data Handling:** Pandas
- **Environment Management:** python-dotenv
- **Programming language:** Python

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Nilkantg/GenAI-powered-Supply-Chain-Assistant.git]
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the root directory of the project.
2.  **Add your API keys to the file:**
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    HF_TOKEN="your_hugging_face_token_here"
    ```

### Running the Application

1.  **Initial Run:**
    The first time you run the script, it will create and populate the ChromaDB vector store. This may take a few minutes.
    ```bash
    python src/main.py
    ```
2.  **Subsequent Runs:**
    On all future runs, the script will detect the existing vector store and load it instantly, making the application much faster to start.

### Using the Assistant

When the script starts, it will prompt you to enter a query. You can ask a variety of questions:
- **Simple Data Retrieval:** "What are the stock levels for SKU1?"
- **Calculations:** "What is the EOQ for SKU2?"
- **Report Generation:** "Generate a report for SKU3."

The assistant will use its tools and knowledge base to provide a detailed and relevant response. If you ask for a report, a Markdown file will be generated and saved in the project's root directory.

```📁 Project Structure

GenAI-Powered-Supply-Chain-Assistant/
├── src/
│   ├── data_loaders/
│   │   └── text_splitter.py
│   ├── Embedding_layer/
│   │   └── embedding.py
│   ├── Model_building/
│   │   ├── groq_model.py
│   │   └── report_generator.py
│   ├── Tools/
│   │   └── tools2.py
│   ├── Vector_Store_db/
│   │   └── vectorstore.py
│   ├── main.py
│   └── user_query.py
├── Datasets/
│   └── Supply_data_1/
│       └── supply_chain_forecast.csv
├── .env
├── README.md
└── requirements.txt 

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📄 License
This project is licensed under the MIT License.