import json
import requests
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, FastAPI

router = APIRouter()
router=APIRouter()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def extract_product_llm(query: str):

    prompt = f"""
You are an information extraction system.

Task:
- Extract the complete product name from the user's query.
- A product is a specific purchasable item.
- Return only the FULL and COMPLETE product name exactly as intended.
- Correct VERY MINUTE TYPOS ONLY.
- DO NOT GUESS AT ALL.
- If no clear product name can be extracted, STRICTLY RETURN null.

Output rules:
- Return ONLY valid JSON
- No explanations
- Use exactly these fields:
  - product_name: string or null
  - status: boolean

Examples:

Input: "Semsoong galaxy ohultra"
Output:
{{"product_name":"Null","status":false}}

Input: "samseng galaxy ulltra"
Output:
{{"product_name":samsung galaxy ultra,"status":true}}

User query:
"{query}"
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0
                }
            }
        )

        data = response.json()
        result_dict = json.loads(data["response"])

        return result_dict.get("product_name"), result_dict.get("status", False)

    except Exception as e:
        print(f"LLM Error: {e}")
        return None, False


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    result: bool
    extracted: Optional[str]


@router.post("/llm", response_model=QueryResponse)
def llm_process(request: QueryRequest):

    query = request.query.strip()

    extracted, result = extract_product_llm(query)

    return QueryResponse(
        result=result,
        extracted=extracted
    )
