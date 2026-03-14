import os
from google import genai
import json
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from fastapi import APIRouter
router = APIRouter()

load_dotenv()
api_key = os.getenv("api_key")
client = genai.Client(api_key=api_key)
def extract_product_llm(query: str): 
    prompt = f"""
    You are an information extraction system.

Task:
- Extract the complete product name from the user's query.
- A product is a specific purchasable item (electronics, appliances, software, hardware, accessories, etc.).
- Return only the FULL and COMPLETE product name exactly as intended in the query.
- If the query contains spelling mistakes, you may correct very small, obvious typos only when the intended product is clearly identifiable.
- Do NOT make large assumptions or guess products when uncertain.
- Do NOT return partial names. Only return the complete product name.
- If no clear product name can be extracted, return null.

Output rules (VERY IMPORTANT):
- Return ONLY valid JSON
- No explanations
- No markdown
- Use exactly these fields:
  - extracted: string or null

If a product name is found:
{{
  "extracted": "<complete product name>"
}}

If no product name is found:
{{
  "extracted": null
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
        return result_dict.get("extracted")

    except Exception as e:
        print(f"LLM Error: {e}")
        return None

class QueryRequest(BaseModel):
    query: str
class QueryResponse(BaseModel):
    extracted:Optional[str]


@router.post("/llm", response_model=QueryResponse)
def llm_process(request: QueryRequest):
    query = request.query.strip().lower()
    extracted = extract_product_llm(query)
    
    return QueryResponse(
        extracted=extracted
    )