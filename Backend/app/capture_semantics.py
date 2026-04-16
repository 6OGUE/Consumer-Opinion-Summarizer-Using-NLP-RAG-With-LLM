import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from typing import List, Dict
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

router = APIRouter()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

_executor = ThreadPoolExecutor(max_workers=8)


# ✅ MODELS (aligned with processing + scoring)
class CommentResult(BaseModel):
    summary: str
    sentiments: Dict[str, str]
    overall_sentiment: str


class ProcessingResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: Dict[int, CommentResult]


class RedditResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: List[Dict]


# 🔥 STRICT PROMPT
STRICT_PROMPT = """You are an expert system for analyzing product reviews.

TASK:
1. Decide if the comment is a "review" or "not review".
2. If it is a review, assign sentiment for EACH given aspect.
        A "review" is ANY comment where the user expresses:
        - personal experience
        - opinion
        - satisfaction/dissatisfaction
        - comparison
        - recommendation

        Even short or indirect opinions count.
3. Also assign ONE overall_sentiment.

STRICT RULES:
- You MUST return ALL aspects provided.
- You MUST NOT add new aspects.
- If an aspect is NOT mentioned → "neutral".
- DO NOT guess or assume sentiment.
- Sentiment values must be EXACTLY one of:
  "positive", "neutral", "negative"
- If unclear → "neutral"
- Output STRICT JSON ONLY. NO TEXT.

OUTPUT FORMAT:

If NOT review:
{{"classification": "not review"}}

If review:
{{
  "classification": "review",
  "sentiments": {{
    "aspect1": "positive",
    "aspect2": "neutral"
  }},
  "overall_sentiment": "positive"
}}

---

EXAMPLES:

Product: "iPhone 15"
Aspects: ["battery","camera","performance","display"]

Comment: "Battery is amazing but camera is average"
Response:
{{
  "classification": "review",
  "sentiments": {{
    "battery": "positive",
    "camera": "neutral",
    "performance": "neutral",
    "display": "neutral"
  }},
  "overall_sentiment": "positive"
}}

Comment: "Does it support fast charging?"
Response:
{{"classification": "not review"}}

Comment: "Camera is terrible and battery drains fast"
Response:
{{
  "classification": "review",
  "sentiments": {{
    "battery": "negative",
    "camera": "negative",
    "performance": "neutral",
    "display": "neutral"
  }},
  "overall_sentiment": "negative"
}}

---

Now analyze:

Product: "{product}"
Aspects: {aspects}

Comment: "{comment}"

Response:
"""


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    try:
        text = text.encode("raw_unicode_escape").decode("unicode_escape", errors="replace")
    except Exception:
        pass

    text = re.sub(r"(?m)^>+\s*", "", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`{1,3}[\s\S]*?`{1,3}", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if len(text) >= 5 else ""


def _call_ollama_sync(comment_text: str, product: str, aspects: List[str]) -> Dict | None:
    prompt = STRICT_PROMPT.format(
        comment=comment_text.replace('"', "'"),
        product=product,
        aspects=aspects
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None

        return json.loads(match.group())

    except Exception:
        return None


async def classify_comment(comment_text: str, product: str, aspects: List[str]) -> Dict | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _call_ollama_sync,
        comment_text,
        product,
        aspects
    )


async def process_comment(comment: Dict, product: str, aspects: List[str]) -> Dict | None:
    body_keys = ("body", "text", "comment", "content")
    key = next((k for k in body_keys if k in comment), None)

    raw = str(comment.get(key, "")) if key else ""
    cleaned = clean_text(raw)

    if not cleaned:
        return None

    result = await classify_comment(cleaned, product, aspects)

    if result is None or result.get("classification") != "review":
        return None

    raw_sentiments = result.get("sentiments", {})
    overall = result.get("overall_sentiment", "neutral")

    # 🔥 STRICT ENFORCEMENT (CRITICAL FIX)
    sentiments = {
        aspect: raw_sentiments.get(aspect, "neutral")
        for aspect in aspects
    }

    return {
        "summary": cleaned,
        "sentiments": sentiments,
        "overall_sentiment": overall
    }


@router.post("/filter-comments", response_model=ProcessingResponse)
async def filter_comments(payload: RedditResponse):
    if not payload.product.strip():
        raise HTTPException(status_code=400, detail="Product required")

    if not payload.comments:
        raise HTTPException(status_code=400, detail="No comments")

    results = await asyncio.gather(
        *[process_comment(c, payload.product, payload.aspects) for c in payload.comments]
    )

    valid = [r for r in results if r is not None]

    formatted = {
        idx: CommentResult(**item)
        for idx, item in enumerate(valid)
    }

    return ProcessingResponse(
        product=payload.product,
        category=payload.category,
        aspects=payload.aspects,
        count=len(formatted),
        comments=formatted,
    )