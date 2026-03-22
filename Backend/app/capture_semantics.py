from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict
import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests

router = APIRouter()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

ONE_SHOT_PROMPT = """Your job is to classify Reddit comments about a product as either "review" or "not review".

A "review" is any comment where the user shares their personal experience, opinion, or assessment of a product — even briefly. This includes short opinions, complaints, praise, comparisons, or recommendations.
A "not review" is a comment that is a question, joke, meme, off-topic discussion, or contains no personal product experience.

Respond ONLY in this exact JSON format, no other text:
{{"classification": "review"}}
or
{{"classification": "not review"}}

Examples:
Comment: "I've been using this blender for 3 months. Super powerful and quiet, but the lid leaks occasionally."
Response: {{"classification": "review"}}

Comment: "Does anyone know if this works with 220V?"
Response: {{"classification": "not review"}}

Comment: "Bought this last week. Honestly not impressed, feels cheap."
Response: {{"classification": "review"}}

Comment: "Lol same, this subreddit is wild"
Response: {{"classification": "not review"}}

Now classify this comment:
Comment: "{comment}"
Response:"""

_executor = ThreadPoolExecutor(max_workers=1)


class DeduplicationResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]


class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]


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
    text = re.sub(r"(?m)^-{3,}$", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;|&#34;", '"', text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= 5 else ""


def _call_ollama_sync(comment_text: str) -> Dict | None:
    """Blocking Ollama call using requests — runs inside a thread."""
    safe_comment = comment_text.replace('"', "'").replace("\n", " ").strip()
    prompt = ONE_SHOT_PROMPT.format(comment=safe_comment)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 30,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not json_match:
            return None

        return json.loads(json_match.group())

    except Exception:
        return None


async def classify_comment_with_ollama(comment_text: str) -> Dict | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _call_ollama_sync, comment_text)


async def process_comment(comment: Dict) -> Dict | None:
    body_keys = ("body", "text", "comment", "content")
    body_key = next((k for k in body_keys if k in comment), None)
    raw_body = str(comment.get(body_key, "")) if body_key else ""

    # Use light clean for classification to preserve context
    cleaned_body = clean_text(raw_body)

    if not cleaned_body:
        return None

    result = await classify_comment_with_ollama(cleaned_body)

    if result is None or result.get("classification", "").lower() != "review":
        return None

    return comment


@router.post("/filter-comments", response_model=DeduplicationResponse)
async def filter_comments(payload: RedditResponse):
    if not payload.product.strip():
        raise HTTPException(status_code=400, detail="Product name must not be empty.")

    if not payload.comments:
        raise HTTPException(status_code=400, detail="No comments provided.")

    results = await asyncio.gather(*[process_comment(c) for c in payload.comments])
    relevant = [r for r in results if r is not None]

    return DeduplicationResponse(
        product=payload.product,
        count=len(relevant),
        comments=relevant,
    )