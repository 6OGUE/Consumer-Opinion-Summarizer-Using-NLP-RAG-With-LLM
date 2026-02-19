from pydantic import BaseModel
from typing import List, Dict
import re
import warnings
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import logging
from keybert import KeyBERT
from transformers import pipeline as hf_pipeline
from fastapi import APIRouter
router = APIRouter()

BART_MODEL_NAME = "facebook/bart-large-cnn"

logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
print("Loading BART tokenizer...")
bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)

print("Loading BART model...")
bart_model = BartForConditionalGeneration.from_pretrained(
    BART_MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")
bart_model.eval()

print("Loading sentiment model...")
sentiment_pipeline = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
)

print("Loading KeyBERT...")
kw_model = KeyBERT()


class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]


class CommentResult(BaseModel):
    summary: str
    sentiment: str          
    keywords: List[str]


class ProcessingResponse(BaseModel):
    product: str
    count: int
    comments: Dict[int, CommentResult]


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


def summarize_with_bart(text: str, max_target_length: int = 60) -> str:
    device = bart_model.device

    inputs = bart_tokenizer(
        text,
        max_length=1024,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_target_length,
            min_length=25,
            num_beams=4,                        
            length_penalty=1.2,
            no_repeat_ngram_size=4,
            early_stopping=True,
        )

    return bart_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    ).strip()


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    raw = kw_model.extract_keywords(
        text,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return [kw for kw, _score in raw]

SUMMARISE_MIN_LENGTH = 50
SUMMARY_LENGTH_RATIO_THRESHOLD = 0.85

def process_comment(text: str) -> CommentResult:
    cleaned = text

    if not cleaned:
        return CommentResult(summary="", sentiment="neutral", keywords=[])
    
    sentiment = get_sentiment(cleaned)
    keywords = extract_keywords(cleaned) if len(cleaned) > 10 else []

    if len(cleaned) < SUMMARISE_MIN_LENGTH:
        summary = cleaned
    else:
        try:
            candidate = summarize_with_bart(cleaned)
            if len(candidate) >= len(cleaned) * SUMMARY_LENGTH_RATIO_THRESHOLD:
                summary = cleaned
            else:
                summary = candidate
        except Exception as e:
            print(f"Summarization error: {e}")
            summary = cleaned

    return CommentResult(
        summary=summary,
        sentiment=sentiment,
        keywords=keywords,
    )

@router.post("/process_comments", response_model=ProcessingResponse)
async def clean_comments(
    reddit_data: RedditResponse,
    comment_key: str = "comment",
):
    results: Dict[int, CommentResult] = {}

    for idx, comment in enumerate(reddit_data.comments):
        original_text = (
            comment.get(comment_key, "")
            if isinstance(comment, dict)
            else str(comment)
        )
        results[idx] = process_comment(original_text)

    return ProcessingResponse(
        product=reddit_data.product,
        count=reddit_data.count,
        comments=results,
    )