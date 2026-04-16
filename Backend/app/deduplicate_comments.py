import re
import numpy as np
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

router = APIRouter()

model = SentenceTransformer('all-MiniLM-L6-v2')


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


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-10)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    b_normalized = b / b_norms
    return b_normalized @ a


class ProductComments(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: List[Dict[str, str]]


class DeduplicationResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: List[Dict[str, str]]


@router.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_comments(
    data: ProductComments,
    similarity_threshold: float = 0.85,
):
    try:
        if not data.comments:
            return DeduplicationResponse(
                product=data.product,
                category=data.category,
                aspects=data.aspects,
                count=0,
                comments=[]
            )

        processed_comments = []
        for c in data.comments:
            cleaned = clean_text(c.get("comment", ""))
            if cleaned:
                processed_comments.append(cleaned)

        if not processed_comments:
            return DeduplicationResponse(
                product=data.product,
                category=data.category,
                aspects=data.aspects,
                count=0,
                comments=[]
            )

        embeddings = model.encode(processed_comments)

        unique_indices: List[int] = [0]

        for i in range(1, len(embeddings)):
            kept_vectors = embeddings[unique_indices]
            similarities = cosine_similarity_numpy(embeddings[i], kept_vectors)

            if np.max(similarities) < similarity_threshold:
                unique_indices.append(i)

        unique_comments = [{"comment": processed_comments[i]} for i in unique_indices]

        return DeduplicationResponse(
            product=data.product,
            category=data.category,
            aspects=data.aspects,
            count=len(unique_comments),
            comments=unique_comments,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))