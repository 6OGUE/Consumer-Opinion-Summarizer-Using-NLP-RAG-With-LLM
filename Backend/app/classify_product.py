from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi import APIRouter
import httpx

router = APIRouter()

# =======================
# INPUT / OUTPUT MODELS
# =======================

class QueryResponse(BaseModel):
    extracted: Optional[str]


class CategoryResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]


# =======================
# HARDCODED CATEGORIES + ASPECTS
# =======================

CATEGORY_ASPECTS: Dict[str, List[str]] = {
    "mobile": ["battery", "camera", "performance", "display"],
    "laptop": ["performance", "battery", "build quality", "display"],
    "headphones": ["sound quality", "battery", "comfort", "noise cancellation"],
    "smartwatch": ["battery", "fitness tracking", "display", "build quality"],
    "tablet": ["performance", "display", "battery", "portability"],
    "camera": ["image quality", "battery", "lens", "video"],
    "television": ["picture quality", "sound", "smart features", "connectivity"],
    "gaming console": ["performance", "game library", "graphics", "cooling"],
    "refrigerator": ["cooling", "energy efficiency", "storage", "noise"],
    "washing machine": ["cleaning", "efficiency", "noise", "capacity"],
    "None":["None"]
}

VALID_CATEGORIES = list(CATEGORY_ASPECTS.keys())
FALLBACK_CATEGORY = "None"


# =======================
# OLLAMA CONFIG
# =======================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


# =======================
# CATEGORY CLASSIFIER
# =======================

def classify_product(product: str) -> str:
    # Build a numbered list to make selection unambiguous
    numbered = "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(VALID_CATEGORIES))

    prompt = f"""Classify the product into one of these categories:

{numbered}

Product: "{product}"

Reply with only the category name. No punctuation, no explanation."""

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,       
                    "num_predict": 10,      
                },
            },
            timeout=30.0,
        )
        response.raise_for_status()

        raw = response.json().get("response", "").strip().lower()

        # Strip punctuation/quotes the model might add
        raw = raw.strip("\"'.,!?").strip()

        # Direct match
        if raw in VALID_CATEGORIES:
            return raw

        # Partial match — handles cases like "mobile phone" → "mobile"
        for cat in VALID_CATEGORIES:
            if cat in raw:
                return cat

        return FALLBACK_CATEGORY

    except Exception as e:
        print(f"Ollama classification error: {e}")
        return FALLBACK_CATEGORY


# =======================
# MAIN API
# =======================

@router.post("/classify_product", response_model=CategoryResponse)
async def classify(query: QueryResponse):
    product = query.extracted or ""

    if not product:
        return CategoryResponse(
            product="",
            category=FALLBACK_CATEGORY,
            aspects=CATEGORY_ASPECTS[FALLBACK_CATEGORY]
        )

    category = classify_product(product)
    aspects = CATEGORY_ASPECTS[category]  # always safe, fallback guarantees a valid key

    return CategoryResponse(
        product=product,
        category=category,
        aspects=aspects
    )