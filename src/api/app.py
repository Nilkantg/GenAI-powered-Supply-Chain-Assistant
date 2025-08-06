from flask import Flask, request, jsonify, render_template
import os
import sys
from dotenv import load_dotenv
import warnings

# --- Path Setup ---
# This is the crucial part: Add the 'src' directory to Python's path
# so it can find all your custom modules.
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, SRC_PATH)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
load_dotenv()

# --- Import Your Custom Modules ---
# These imports assume all your .py files are in the same directory as this app.py
from data_loaders.text_splitter import load_data, LoadedData
from Embedding_layer.embedding import embedding_, DataLoader
from Vector_Store_db.vectorstore import create_vector_store, Vector_loader
from Model_building.groq_model import build_model, Model
from Tools.tools2 import tool_binding
from user_query import query_user_input, UserQueryInput

# --- Flask App Initialization ---
# Tell Flask where to find the 'templates' folder.
app = Flask(__name__, template_folder=os.path.join(SRC_PATH, 'templates'))

# --- Global Variable for the Agent ---
# This will hold the initialized model and tools so they don't reload on every request.
agent_payload = {}

# --- One-Time Initialization Function ---
def initialize_supply_chain_agent():
    """
    This function runs once before the first request to load data,
    create embeddings, build the vector store, and prepare the agent.
    It uses the functions from your provided Python files.
    """
    global agent_payload
    print("Initializing Supply Chain Agent... This may take a moment.")

    # --- Step 1: Check for API Keys ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not GROQ_API_KEY or not HF_TOKEN:
        raise ValueError("GROQ_API_KEY and/or HF_TOKEN not found in environment variables. Please check your .env file.")

    # --- Step 2: Load Data using your text_splitter.py ---
    file_path = "Datasets/Supply_data_1/supply_chain_forecast.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The data file was not found at {file_path}. Please ensure it's in the correct directory.")
        
    data_loader_input = LoadedData(file_path=file_path)
    df = load_data(data_loader_input)
    print("Data loaded successfully.")

    # --- Step 3: Prepare Documents and Metadata ---
    text_documents = df.apply(lambda row: ", ".join(row.astype(str)), axis=1).tolist()
    metadatas = [{"sku": row["SKU"]} for _, row in df.iterrows()]

    # --- Step 4: Create Embeddings using your embedding.py ---
    embedding_input = DataLoader(HF_TOKEN=HF_TOKEN, text=text_documents)
    embedding_vectors = embedding_(embedding_input)
    print("Embeddings created.")

    # --- Step 5: Create Vector Store using your vectorstore.py ---
    vector_loader_input = Vector_loader(vectors=embedding_vectors, texts=text_documents, metadatas=metadatas)
    vector_db = create_vector_store(vector_loader_input)
    print("Vector store created.")

    # --- Step 6: Bind Tools using your tools2.py ---
    # The tool_binding function requires the vector_store for the report generator.
    agent_tools = tool_binding(groq_api_key=GROQ_API_KEY, vector_store=vector_db)
    print("Tools have been bound.")

    # --- Step 7: Store the necessary components for later use ---
    agent_payload = {
        "groq_api_key": GROQ_API_KEY,
        "vector_store": vector_db,
        "tools": agent_tools
    }
    print("Initialization complete. The application is ready.")


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page from the 'src/templates' folder."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    """
    Handles the user query from the frontend by calling your build_model function.
    """
    global agent_payload
    
    json_data = request.get_json()
    query_text = json_data.get('query')

    if not query_text:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        # --- Use your user_query.py component ---
        user_query = UserQueryInput(query=query_text)
        processed_query = query_user_input(user_query)

        # --- Build the model input using your groq_model.py schema ---
        model_input = Model(
            query=processed_query,
            groq_api_key=agent_payload['groq_api_key'],
            vector_store=agent_payload['vector_store'],
            tools=agent_payload['tools']
        )

        # --- Run the model using your groq_model.py function ---
        result = build_model(model_input)
        
        response = result.get('output', 'Sorry, I could not process that request.')
        return jsonify({"response": response})

    except Exception as e:
        print(f"An error occurred during model execution: {e}")
        # It's useful to log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while processing the request."}), 500

# --- Application Startup ---
# This block ensures that the initialization runs only once when the server starts.
with app.app_context():
    initialize_supply_chain_agent()

if __name__ == '__main__':
    # Setting debug=False is recommended for production, but True is fine for development.
    # The reloader can sometimes cause initialization to run twice in debug mode.
    app.run(host='0.0.0.0', port=5001, debug=False)
