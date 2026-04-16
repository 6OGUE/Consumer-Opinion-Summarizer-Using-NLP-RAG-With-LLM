from pydantic import BaseModel
from typing import List, Dict
from fastapi import APIRouter
import httpx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

router = APIRouter()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
SUMMARISE_MIN_LENGTH = 200

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = 0 if torch.cuda.is_available() else -1
if DEVICE == 0:
    model = model.cuda()


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


class FilterResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: Dict[int, CommentResult]


# 🔹 Summarization
def shorten_with_ollama(text: str) -> str:
    prompt = (
        "Shorten the following comment to max 200 characters while preserving all key product-related details. "
        "RETURN ONLY THE SHORTENED COMMENT.\n"
        f"Comment: {text}"
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
        return response.json().get("response", "").strip() or text
    except Exception:
        return text


# 🔹 DeBERTa overall sentiment (proper usage)
def get_overall_sentiment(text: str, product: str) -> str:
    try:
        query = f"What is the overall sentiment towards {product}?"

        inputs = tokenizer(
            query,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        if DEVICE == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        scores = F.softmax(outputs.logits, dim=-1)

        confidence = torch.max(scores).item()
        label_idx = torch.argmax(scores, dim=-1).item()

        if confidence < 0.5:
            return "neutral"

        return {0: "negative", 1: "neutral", 2: "positive"}.get(label_idx, "neutral")

    except Exception:
        return "neutral"


def process_comment(comment: CommentResult, product: str) -> CommentResult:
    text = comment.summary

    summary = shorten_with_ollama(text) if len(text) >= SUMMARISE_MIN_LENGTH else text

    overall = get_overall_sentiment(summary, product)

    return CommentResult(
        summary=summary,
        sentiments=comment.sentiments,
        overall_sentiment=overall            
    )


@router.post("/process_comments", response_model=ProcessingResponse)
async def process_comments(data: FilterResponse):
    results: Dict[int, CommentResult] = {}

    for idx, comment in data.comments.items():
        results[idx] = process_comment(comment, data.product)

    return ProcessingResponse(
        product=data.product,
        category=data.category,
        aspects=data.aspects,
        count=data.count,
        comments=results,
    )