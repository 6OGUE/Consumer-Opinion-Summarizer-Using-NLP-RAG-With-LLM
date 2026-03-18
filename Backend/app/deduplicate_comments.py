from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import APIRouter

router = APIRouter()

model = SentenceTransformer('all-MiniLM-L6-v2')

class Comment(BaseModel):
    comment: str

class ProductComments(BaseModel):
    product: str
    count: int
    comments: List[Dict[str, str]]


class DeduplicationResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict[str, str]]


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-10)                          
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10   
    b_normalized = b / b_norms                                     
    return b_normalized @ a                                        


@router.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_comments(
    data: ProductComments,
    similarity_threshold: float = 0.55,
):
    try:
        if not data.comments:
            return DeduplicationResponse(product=data.product, count=0, comments=[])

        comment_texts = [c["comment"] for c in data.comments]
        embeddings = model.encode(comment_texts)  

        
        unique_indices: List[int] = [0]  

        for i in range(1, len(embeddings)):
            kept_vectors = embeddings[unique_indices]  
            similarities = cosine_similarity_numpy(embeddings[i], kept_vectors)

        
            if np.max(similarities) < similarity_threshold:
                unique_indices.append(i)

        unique_comments = [{"comment": comment_texts[i]} for i in unique_indices]

        return DeduplicationResponse(
            product=data.product,
            count=len(unique_comments),
            comments=unique_comments,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))