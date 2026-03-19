from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

class DeduplicationResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict[str, str]]


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
    text = re.sub(r"&quot;|&#34;", "", text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r'[\"\'`\*#\[\](){}<>|\\^~@$%]', " ", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= 5 else ""

_ANCHOR_TEMPLATES = [
    "I bought {product} and here is my honest review.",
    "My experience using {product} has been good and bad in these ways.",

    "{product} performs well in these areas and struggles in these areas.",
    "The build quality and design of {product} is impressive or disappointing.",
    "Battery life, speed, and reliability of {product} compared to expectations.",

    "I would recommend {product} to others for these reasons.",
    "I would not recommend {product} because of these problems.",
    "{product} is worth buying or not worth buying at this price.",

    "{product} is better or worse than its competitors in these ways.",
    "Compared to similar products, {product} stands out or falls short.",

    "The biggest problem I have with {product} is this specific issue.",
    "The best thing about {product} is this specific feature or aspect.",
    "Customer support and after-sales experience with {product}.",

    "{product} is great for this type of person or this use case.",
    "{product} is not suitable for this type of person or this use case.",
]


def build_query_embedding(product: str, model: SentenceTransformer) -> np.ndarray:

    anchors = [t.format(product=product) for t in _ANCHOR_TEMPLATES]
    anchor_embeddings = model.encode(anchors, normalize_embeddings=True, show_progress_bar=False)
    mean_vec = anchor_embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    return (mean_vec / norm).reshape(1, -1)  

@router.post("/filter-comments", response_model=DeduplicationResponse)
async def filter_comments(payload: RedditResponse, threshold: float = 0.5):
    if not payload.product.strip():
        raise HTTPException(status_code=400, detail="Product name must not be empty.")

    if not payload.comments:
        raise HTTPException(status_code=400, detail="No comments provided.")

    model = get_model()
    query_emb = build_query_embedding(payload.product.strip(), model)

    body_keys = ("body", "text", "comment", "content")
    cleaned_meta: List[Dict[str, str]] = []
    texts_to_embed: List[str] = []

    for comment in payload.comments:
        body_key = next((k for k in body_keys if k in comment), None)
        raw_body = str(comment.get(body_key, "")) if body_key else ""
        cleaned_body = clean_text(raw_body)

        if not cleaned_body:
            continue

        cleaned = {k: clean_text(str(v)) for k, v in comment.items()}
        cleaned[body_key or "body"] = cleaned_body

        texts_to_embed.append(cleaned_body)
        cleaned_meta.append(cleaned)

    if not texts_to_embed:
        return DeduplicationResponse(product=payload.product, count=0, comments=[])

    comment_embeddings = model.encode(
        texts_to_embed,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )

    relevant: List[Dict[str, str]] = []
    similarities = cosine_similarity(comment_embeddings, query_emb).flatten()

    for sim, meta in zip(similarities, cleaned_meta):
        if float(sim) >= threshold:
            relevant.append(meta)

    return DeduplicationResponse(
        product=payload.product,
        count=len(relevant),
        comments=relevant,
    )