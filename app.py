import pandas as pd
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import os
import numpy as np

# --- ⚠️ IMPORTANT: Set your Gemini API Key ---
# For security, you should set this as an environment variable (e.g., in .env)
# But for a simple demo, you can replace 'YOUR_API_KEY' with your actual key string.
# The code will first try to use the GEMINI_API_KEY environment variable.
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBB1sm7BT1WnpkbwHlqehp4I3ADNrLv83s")
if API_KEY == "AIzaSyBB1sm7BT1WnpkbwHlqehp4I3ADNrLv83s":
    print("WARNING: Using placeholder API key. Set the GEMINI_API_KEY environment variable or replace the placeholder in app.py.")
# ---------------------------------------------

# Constants for column names
COL_ITEM_NAME = 'subsubcategory'
COL_PRICE = 'avg_price'
COL_LOCATION = 'location'
COL_UNIT = 'unit_raw' 

app = Flask(__name__)

# Initialize the Gemini Client
try:
    gemini_client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_client = None


def load_item_data():
    """Loads, cleans, and aggregates data from the Parquet file."""
    parquet_path = "Avg Price PI Aug25 (location).parquet"
    try:
        df = pd.read_parquet(parquet_path)

        # Drop rows where the item name (subsubcategory) is missing or empty/NaN.
        df.dropna(subset=[COL_ITEM_NAME, COL_LOCATION, COL_UNIT, COL_PRICE], inplace=True)
        df = df[df[COL_ITEM_NAME].astype(str).str.strip() != '']

        # Group by location, unit, and item name, then calculate the average price
        grouped_df = df.groupby([COL_LOCATION, COL_UNIT, COL_ITEM_NAME])[COL_PRICE].mean().reset_index()

        # Standardize names for JavaScript
        grouped_df.columns = ['location', 'unit', 'name', 'price']
        
        # Format the price to two decimal places for cleaner output
        grouped_df['price'] = grouped_df['price'].round(2)

        # Extract unique lists for dropdowns
        unique_locations = sorted(grouped_df['location'].unique().tolist())
        unique_units = sorted(grouped_df['unit'].unique().tolist())
        
        # Convert aggregated data to a list of dictionaries
        item_data_list = grouped_df.to_dict('records')

        print(f"Successfully loaded {len(item_data_list)} unique item combinations from Parquet.")
        return item_data_list, unique_locations, unique_units

    except FileNotFoundError:
        print(f"Error: Parquet file not found at {parquet_path}. Using empty data.")
        return [], [], []
    except KeyError as e:
        print(f"Error: Missing column in Parquet file. Details: {e}. Using empty data.")
        return [], [], []
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}. Using empty data.")
        return [], [], []

# Load data once when the app starts.
PI_DATA, UNIQUE_LOCATIONS, UNIQUE_UNITS = load_item_data()


@app.route('/')
def index():
    """Renders the main budget calculator page."""
    # This route now serves the single index.html containing the widget
    return render_template(
        'index.html', 
        item_data=PI_DATA, 
        locations=UNIQUE_LOCATIONS, 
        units=UNIQUE_UNITS
    )

@app.route('/chatbot')
def chatbot_page():
    """Renders the dedicated chatbot page."""
    return render_template('chatbot.html')


@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handles chatbot queries using the Gemini API and the loaded item data."""
    if not gemini_client:
        return jsonify({'response': "AI Client not initialized. Please check your API key and setup."}), 500

    user_prompt = request.json.get('prompt')
    if not user_prompt:
        return jsonify({'response': "No prompt provided."}), 400

    # Convert the list of item dictionaries to a string format that Gemini can easily parse and reason over.
    # Using JSON string is highly effective for grounding the model in specific data.
    item_data_json = pd.DataFrame(PI_DATA).to_json(orient='records')
    
    # 1. Define the System Instruction for context and role
    system_instruction = (
        "You are an expert Price Intelligent (PI) assistant. Your task is to analyze the provided "
        "JSON data containing item names, locations, units, and average prices in RM, and answer "
        f"the user's question based *only* on this data. The data has {len(PI_DATA)} entries."
        "Provide specific item names, prices, locations, and units from the data to support your answer. "
        "Do not invent information. Format the final response clearly, using bold text for key results."
    )

    # 2. Construct the full prompt
    full_prompt = (
        f"Data for analysis (JSON Array of Objects):\n"
        f"```json\n{item_data_json}\n```\n\n"
        f"User Question:\n{user_prompt}"
    )

    try:
        # 3. Call the Gemini API
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        return jsonify({'response': response.text})

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({'response': f"Sorry, there was an error processing your request: {e}"}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 for development if needed for external access, otherwise use 127.0.0.1
    app.run(debug=True, host='127.0.0.1', port=5000)