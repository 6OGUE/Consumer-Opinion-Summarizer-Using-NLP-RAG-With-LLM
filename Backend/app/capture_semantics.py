from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
from typing import List, Dict
import requests
import json

router=APIRouter()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

class DeduplicationResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict[str, str]]


class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]


def build_prompt(data: DeduplicationResponse) -> str:
    return f"""
You are a strict JSON processor.

Your job is to CLEAN and FILTER Reddit comments about the product: "{data.product}"

DEFINITIONS:

RELEVANT comment:
- Mentions the product directly, OR
- Mentions features of the product.


OUTPUT RULES (STRICT):

1. STRICTLY RETURN ONLY the semantic meaning relevant comments, Discard everything else

OUTPUT FORMAT (VERY STRICT JSON ONLY):

{{
  "product": string,
  "count": number,
  "comments": [
    {{"comment": string}}
  ]
}}

Return ONLY valid JSON. NOTHING ELSE.
Input data:
Product: {data.product}
Count: {data.count}
Comments:
{json.dumps(data.comments, indent=2)}
"""


def call_ollama(prompt: str) -> dict:

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"  
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Ollama error")

    result = response.json()

    try:
        return json.loads(result["response"])
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid JSON from model")


@router.post("/process", response_model=RedditResponse)
def process_data(data: DeduplicationResponse):

    prompt = build_prompt(data)

    llm_output = call_ollama(prompt)

    return RedditResponse(**llm_output)
