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


def extract_product_llm(query: str) -> Tuple[Optional[str], Optional[str]]:
    prompt = f"""
    You are a strict product name extraction system.

    Your task:
    Extract product names only if a specific commercial product model appears in the user query.

    Rules:

    1. Only return real product names that represent a specific commercial product model.

    2. Never return generic phrases, categories, product types, or descriptive phrases.

    3. A product name must explicitly appear in the query text.

    4. Do not infer or guess products from categories or context.

    5. Suggestions are allowed ONLY if the query contains part of an identifiable product name such as a brand or model token.
    If the query contains only generic category words, suggestions must be null.

    6. If no explicit product name or identifiable product token exists in the query, return null for both fields.

    7. Small typo correction is allowed only when the intended product is clearly identifiable.

    8. Confidence levels:

    FULL MATCH (100% certain):

    * The exact product name appears in the query.
    * Return that name in "extracted".
    * Set "suggestions" to null.

    PARTIAL MATCH:

    * The query contains part of a real product name but not the full model name.
    * Return the complete product name in "suggestions".
    * Set "extracted" to null.

    NO MATCH:

    * If the query contains only generic product categories or unrelated text.
    * Return both fields as null.

    9. Suggestions must always be a single valid product model name.

    10. Before returning output, verify that the returned value is a specific product model name.
    If it is not a specific product model name, return null.

    11. Never return both fields filled.

    12. Never explain anything.

    13. Never guess or generate product names that are not explicitly supported by the query.

    Output format (STRICT JSON ONLY):

    {
    "extracted": string | null,
    "suggestions": string | null
    }

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
        suggestions = result_dict.get("suggestions")

        extracted = extracted.strip().lower() if extracted else None
        suggestions = suggestions.strip().lower() if suggestions else None

        return extracted, suggestions

    except Exception as e:
        print(f"LLM Error: {e}")
        return None, None


def extract_product_dict(query: str) -> Tuple[Optional[str], Optional[str]]:
    if not query:
        return None, None

    normalized_query = query.strip().lower()

    if normalized_query in PRODUCT_DICT:
        return normalized_query, None

    
    for product in PRODUCT_DICT:
        if normalized_query in product:
            return None, product

    
    best_match = None
    for product in PRODUCT_DICT:
        if product in normalized_query:
            if best_match is None or len(product) > len(best_match):
                best_match = product

    if best_match:
        return None, best_match

    return None, None


def validate_and_extract_product(query: str) -> Tuple[Optional[str], Optional[str]]:
    if not query or not query.strip():
        return None, None

    
    extracted, suggestions = extract_product_dict(query)

    if extracted is not None or suggestions is not None:
        return extracted, suggestions

    extracted, suggestions = extract_product_llm(query)
    return extracted, suggestions


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    extracted: Optional[str]
    suggestions: Optional[str]


@router.post("/local", response_model=QueryResponse)
def local_process(request: QueryRequest):
    query = request.query.strip().lower()
    extracted, suggestions = validate_and_extract_product(query)

    return QueryResponse(
        extracted=extracted,
        suggestions=suggestions
    )
