import numpy as np
import re
import requests
from typing import List, Dict
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

router = APIRouter()

TOP_K = 5
MIN_SIMILARITY = 0.25
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ✅ ONLY CHANGE: added category + aspects
class ChatRequest(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: List[Dict]
    question: str


# ✅ ONLY CHANGE: added category + aspects
class ChatResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    question: str
    answer: str


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def retrieve_relevant_sentences(
    question: str,
    comments: List[Dict],
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> List[str]:
    sentences: List[str] = []
    for c in comments:
        text = c.get("comment", str(c))
        sentences.extend(split_into_sentences(text))

    if not sentences:
        return []

    model = get_embedding_model()

    all_embeddings = model.encode(
        [question] + sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = all_embeddings[0]
    sentence_embeddings = all_embeddings[1:]

    scores = cosine_similarity(query_embedding, sentence_embeddings)

    k = min(top_k, len(sentences))
    top_indices = np.argsort(scores)[::-1][:k]
    top_indices = [i for i in top_indices if scores[i] >= min_similarity]

    return [sentences[i] for i in top_indices]


def rephrase_with_ollama(question: str, raw_answer: str) -> str:
    prompt = (
        f"You are a professional assistant analyzing product reviews. "
        f"A user asked the following question:\n"
        f"Question: {question}\n\n"
        f"The following raw information was retrieved from user reviews:\n"
        f"Raw answer: {raw_answer}\n\n"
        f"Rephrase the raw answer into a single, formal, and coherent response that "
        f"directly addresses the question. Always begin your response with 'Based on the reviews, '. "
        f"Do not add information that is not present in the raw answer. "
        f"Reply with only the rephrased answer, nothing else."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", raw_answer).strip()

    except requests.exceptions.ConnectionError:
        return raw_answer
    except requests.exceptions.Timeout:
        return raw_answer
    except Exception:
        return raw_answer


@router.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    relevant_sentences = retrieve_relevant_sentences(
        question=data.question,
        comments=data.comments,
    )

    if not relevant_sentences:
        raw_answer = "I don't have enough information in the provided data to answer that question."
    else:
        raw_answer = " ".join(relevant_sentences)

    answer = rephrase_with_ollama(
        question=data.question,
        raw_answer=raw_answer,
    )

    return ChatResponse(
        product=data.product,
        category=data.category,   # ✅ added
        aspects=data.aspects,     # ✅ added
        question=data.question,
        answer=answer,
    )