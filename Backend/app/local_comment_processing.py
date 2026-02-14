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

URL_PATTERN = re.compile(r'http[s]?://\S+')

REDDIT_PATTERNS = [
    re.compile(r'\[deleted\]'),
    re.compile(r'\[removed\]'),
    re.compile(r'u/\w+'),
    re.compile(r'r/\w+'),
    re.compile(r'!ping\s+\w+'),
    re.compile(r'RemindMe!\s+.*'),
    re.compile(r'^\s*>\s*', re.MULTILINE),
    re.compile(r'^\s*Edit\d*:\s*', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*Update:\s*', re.MULTILINE | re.IGNORECASE),
]

EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

FILLER_PHRASES = [
    'i mean', 'you know', 'like', 'basically', 'literally', 'actually',
    'tbh', 'imo', 'imho', 'fwiw', 'afaik', 'to be honest', 'in my opinion',
    'honestly', 'personally', 'i think that', 'i feel like', 'it seems like',
    'kind of', 'sort of', 'a bit', 'pretty much', 'i guess'
]


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


def strip_reddit_noise(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = URL_PATTERN.sub('', text)
    for pattern in REDDIT_PATTERNS:
        text = pattern.sub('', text)
    text = EMOJI_PATTERN.sub('', text)
    for phrase in FILLER_PHRASES:
        text = re.sub(
            r'\b' + re.escape(phrase) + r'\b',
            '',
            text,
            flags=re.IGNORECASE
        )
    return re.sub(r'\s+', ' ', text).strip()

SUMMARISE_MIN_LENGTH = 50
SUMMARY_LENGTH_RATIO_THRESHOLD = 0.85

def process_comment(text: str) -> CommentResult:
    cleaned = strip_reddit_noise(text)

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
    comment_key: str = "body",
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