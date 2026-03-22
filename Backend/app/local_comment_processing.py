from pydantic import BaseModel
from typing import List, Dict
import warnings
import torch
from transformers import pipeline as hf_pipeline
from transformers import logging
from fastapi import APIRouter
import httpx

router = APIRouter()

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("Loading sentiment model...")
sentiment_pipeline = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
)


class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]


class CommentResult(BaseModel):
    summary: str
    sentiment: str


class ProcessingResponse(BaseModel):
    product: str
    count: int
    comments: Dict[int, CommentResult]


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
SUMMARISE_MIN_LENGTH = 100


def get_sentiment(text: str) -> str:
    LABEL_MAP = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }
    try:
        result = sentiment_pipeline(text)[0]
        return LABEL_MAP.get(result["label"], "neutral")
    except Exception:
        return "neutral"


def shorten_with_ollama(text: str) -> str:
    prompt = (
        "Shorten the following comment to a max of 100 chars while preserving all pros and cons mentioned exactly. "
        "Do not add anything new.Do not shorten too much.Should not contain unnecessary slashes or quotes. Return only the shortened comment, no preamble.\n\n"
        f"Comment:\n{text}"
    )
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        shortened = response.json().get("response", "").strip()
        return shortened if shortened else text
    except Exception as e:
        print(f"Ollama error: {e}")
        return text


def process_comment(text: str) -> CommentResult:
    if not text:
        return CommentResult(summary="", sentiment="neutral")

    sentiment = get_sentiment(text)

    if len(text) >= SUMMARISE_MIN_LENGTH:
        summary = shorten_with_ollama(text)
    else:
        summary = text

    return CommentResult(summary=summary, sentiment=sentiment)


@router.post("/process_comments", response_model=ProcessingResponse)
async def clean_comments(
    reddit_data: RedditResponse,
):
    results: Dict[int, CommentResult] = {}

    for idx, comment in enumerate(reddit_data.comments):
        original_text = (
            comment.get("comment", "")
            if isinstance(comment, dict)
            else str(comment)
        )
        results[idx] = process_comment(original_text)

    return ProcessingResponse(
        product=reddit_data.product,
        count=reddit_data.count,
        comments=results,
    )