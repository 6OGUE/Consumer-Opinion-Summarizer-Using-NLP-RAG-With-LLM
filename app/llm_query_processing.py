import os
from google import genai
import json
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("api_key")
client = genai.Client(api_key)
def extract_product_llm(query: str): # pass query as param
    prompt = f"""
    You are an information extraction system.
    Task:
    - Extract a product name from the user's query.
    - A product is a specific purchasable item (electronics, appliances, software, etc.).
    Output rules (VERY IMPORTANT):
    - Return ONLY valid JSON
    - No explanations, no markdown
    - Use exactly these fields:
    - product_name: string or null
    - status: boolean
    If a product name is found:
    {{
    "product_name": "<product name>",
    "status": true
    }}
    If no product name is found:
    {{
    "product_name": null,
    "status": false
    }}

    User query:
    "{query}"
    """

    try:
        response = client.models.generate_content(
        model="gemini-2.5-flash-lite", 
        contents=prompt,
        config={
        "temperature": 0,
        "max_output_tokens": 100,
        "response_mime_type": "application/json"
        })
        result_dict = json.loads(response.text)
        return result_dict.get("product_name"), result_dict.get("status", False)

    except Exception as e:
        print(f"LLM Error: {e}")
        return None, False

product_name, status = extract_product_llm(query)
print("Result -", status)
print("Extracted -", product_name)