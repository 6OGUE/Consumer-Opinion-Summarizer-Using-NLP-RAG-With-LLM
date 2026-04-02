from pydantic import BaseModel
from typing import List, Dict
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import logging
from fastapi import APIRouter
import httpx

router = APIRouter()

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("Loading ABSA sentiment model...")
ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
absa_tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_NAME)
absa_model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_NAME)
absa_model.eval()

DEVICE = 0 if torch.cuda.is_available() else -1
if DEVICE == 0:
    absa_model = absa_model.cuda()


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
SUMMARISE_MIN_LENGTH = 200


def get_sentiment(text: str, product: str) -> str:
    try:
        inputs = absa_tokenizer(
            product,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        if DEVICE == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = absa_model(**inputs)

        scores = F.softmax(outputs.logits, dim=-1)
        label_idx = torch.argmax(scores, dim=-1).item()

        LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
        return LABEL_MAP.get(label_idx, "neutral")

    except Exception as e:
        print(f"ABSA sentiment error: {e}")
        return "neutral"


def shorten_with_ollama(text: str) -> str:
    prompt = (
        "Shorten the following comment to a max of 200 chars while preserving all pros, cons and important product related details mentioned exactly. "
        "Do not add anything new.Do not shorten too much.Should not contain unnecessary slashes or quotes. RETURN ONLY THE SHORTENED COMMENT, NOTHING ELSE."
        f"Comment:{text}"
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


def process_comment(text: str, product: str) -> CommentResult:
    if not text:
        return CommentResult(summary="", sentiment="neutral")

    sentiment = get_sentiment(text, product)

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
        results[idx] = process_comment(original_text, reddit_data.product)

    return ProcessingResponse(
        product=reddit_data.product,
        count=reddit_data.count,
        comments=results,
    )