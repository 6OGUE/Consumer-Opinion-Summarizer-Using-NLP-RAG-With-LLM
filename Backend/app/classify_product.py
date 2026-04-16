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
}

VALID_CATEGORIES = list(CATEGORY_ASPECTS.keys())


# =======================
# OLLAMA CONFIG
# =======================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


# =======================
# CATEGORY CLASSIFIER
# =======================

def classify_product(product: str) -> str:
    prompt = f"""
You are a strict product classifier.

Choose ONLY ONE category from the list below:

{", ".join(VALID_CATEGORIES)}

Product: {product}

Rules:
- Return ONLY the category name
- No explanation
- No extra text
- Must match exactly from list
"""

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=30.0,
        )

        response.raise_for_status()
        category = response.json().get("response", "").strip().lower()

        # Safety check
        if category not in VALID_CATEGORIES:
            return "mobile"  # fallback default

        return category

    except Exception as e:
        print(f"Ollama classification error: {e}")
        return "mobile"  # fallback


# =======================
# MAIN API
# =======================

@router.post("/classify_product", response_model=CategoryResponse)
async def classify(query: QueryResponse):

    product = query.extracted or ""

    if not product:
        return CategoryResponse(
            product="",
            category="mobile",
            aspects=CATEGORY_ASPECTS["mobile"]
        )

    category = classify_product(product)
    aspects = CATEGORY_ASPECTS.get(category, CATEGORY_ASPECTS["mobile"])

    return CategoryResponse(
        product=product,
        category=category,
        aspects=aspects
    )