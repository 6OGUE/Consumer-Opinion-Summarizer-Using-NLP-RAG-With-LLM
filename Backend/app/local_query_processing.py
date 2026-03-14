import warnings
import os
import logging
import json
import requests

from typing import Optional, Tuple
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()
#################### Cleaner terminal output #########################
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.convert_slow_tokenizer"
)

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

PRODUCT_DICT: set[str] = set()

def load_product_dictionary(file_path="products.txt"):
    global PRODUCT_DICT
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            PRODUCT_DICT = set(line.strip().lower() for line in f if line.strip())
        print(f"Loaded {len(PRODUCT_DICT)} products from {file_path}")
    except FileNotFoundError:
        print(f"Product dictionary file '{file_path}' not found.")

load_product_dictionary()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def extract_product_llm(query: str) -> Optional[str]:
    prompt = f"""
    You are a strict product name extraction system.

    Your task:
    Extract a specific product name from the user query, or return null if no specific product can be confidently identified.

    Rules:

    1. SPECIFIC PRODUCT ONLY:
       A valid extracted value must be a specific, real-world commercial product model.
       It must have enough identifiers to distinguish it from other products.
       Vague or incomplete references do not qualify.

    2. TYPO CORRECTION:
       If the query contains misspellings but the intended product is unambiguously identifiable,
       return the correctly spelled product name.
       If the typos are too severe to confidently identify the product, return null.

    3. NEVER RETURN GENERIC TERMS:
       Do not return product categories, types, adjectives, or descriptive phrases.
       A product name is NOT valid if it could describe a broad class of items rather than one specific model.

    4. CONFIDENCE REQUIREMENT:
       Only return a value if you are confident it refers to one specific product.
       If the query is ambiguous, incomplete, or could refer to multiple different products, return null.

    5. NO INVENTION:
       Never generate, guess, or construct a product name that is not clearly supported by the query text.
       The returned name must be traceable back to tokens present in the query.

    6. CORRECTNESS OVER COMPLETENESS:
       It is better to return null than to return a wrong, generic, or uncertain value.

    7. Never explain anything.

    Output format (STRICT JSON ONLY):

    {{
    "extracted": string | null
    }}

    User query: "{query}"
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0}
            },
            timeout=30
        )
        data = response.json()
        result_dict = json.loads(data.get("response", "{}"))

        extracted = result_dict.get("extracted")
        extracted = extracted.strip().lower() if extracted else None

        return extracted

    except Exception as e:
        print(f"LLM Error: {e}")
        return None


def extract_product_dict(query: str) -> Optional[str]:
    if not query:
        return None

    normalized_query = query.strip().lower()

    for item in PRODUCT_DICT:
        if item==normalized_query:
            return item

    return None


def validate_and_extract_product(query: str) -> Optional[str]:
    if not query or not query.strip():
        return None

    extracted = extract_product_dict(query)

    if extracted is not None:
        return extracted

    extracted = extract_product_llm(query)
    return extracted


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    extracted: Optional[str]


@router.post("/local", response_model=QueryResponse)
def local_process(request: QueryRequest):
    query = request.query.strip().lower()
    extracted = validate_and_extract_product(query)

    return QueryResponse(extracted=extracted)