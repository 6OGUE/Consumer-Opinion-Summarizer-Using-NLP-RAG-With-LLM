from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
from typing import List, Dict
import requests
import json

router=APIRouter()

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

1. STRICTLY Keep ONLY the semantic meaning relevant comments, Discard everything else
2. Clean invalid characters
3. Extract only semantic meaning preserving keywords
4. Each comment must contain meaningful product-related information
5. DISCARD comments that become empty after cleaning
6. DO NOT include author names
7. DO NOT include irrelevant comments

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "product": string,
  "count": number,
  "comments": [
    {{"comment": string}}
  ]
}}

Return ONLY valid JSON. No explanation.
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
