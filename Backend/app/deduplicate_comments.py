from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json


def cosine_similarity_numpy(a, b):
    """Calculate cosine similarity between vector a and matrix b"""
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / np.linalg.norm(a)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    b_norm = b / b_norms
    return np.dot(b_norm, a_norm)

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

VECTOR_DB_PATH = "vector_db.npy"
METADATA_DB_PATH = "metadata_db.json"


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


def load_vector_db():
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(METADATA_DB_PATH):
        vectors = np.load(VECTOR_DB_PATH)
        with open(METADATA_DB_PATH, 'r') as f:
            metadata = json.load(f)
        return vectors, metadata
    return np.array([]), []


def save_vector_db(vectors, metadata):
    if len(vectors) > 0:
        np.save(VECTOR_DB_PATH, vectors)
        with open(METADATA_DB_PATH, 'w') as f:
            json.dump(metadata, f)


def clear_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        os.remove(VECTOR_DB_PATH)
    if os.path.exists(METADATA_DB_PATH):
        os.remove(METADATA_DB_PATH)

async def embed_comments(data: ProductComments):
    try:
        existing_vectors, existing_metadata = load_vector_db()
        comment_texts = [comment['comment'] for comment in data.comments]
        embeddings = model.encode(comment_texts)
        
        new_metadata = [
            {
                "product": data.product,
                "comment": comment['comment']
            }
            for comment in data.comments
        ]
        
        if len(existing_vectors) > 0:
            combined_vectors = np.vstack([existing_vectors, embeddings])
            combined_metadata = existing_metadata + new_metadata
        else:
            combined_vectors = embeddings
            combined_metadata = new_metadata
        
        save_vector_db(combined_vectors, combined_metadata)
        
        return {
            "status": "success",
            "message": f"Embedded {len(comment_texts)} comments",
            "total_comments_in_db": len(combined_metadata)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_comments(data: ProductComments, similarity_threshold: float = 0.70):
    try:
        await embed_comments(data)

        vectors, metadata = load_vector_db()
        
        if len(vectors) == 0:
            clear_vector_db()
            return DeduplicationResponse(
                product=data.product,
                count=0,
                comments=[]
            )
        
        product_indices = [i for i, meta in enumerate(metadata) if meta['product'] == data.product]
        
        if len(product_indices) == 0:
            clear_vector_db()
            return DeduplicationResponse(
                product=data.product,
                count=0,
                comments=[]
            )
        
        product_vectors = vectors[product_indices]
        product_metadata = [metadata[i] for i in product_indices]
        
        unique_indices = []
        unique_indices.append(0)  
        
        for i in range(1, len(product_vectors)):
            similarities = cosine_similarity_numpy(
                product_vectors[i],
                product_vectors[unique_indices]
            )

            if np.max(similarities) < similarity_threshold:
                unique_indices.append(i)
        
        unique_comments = [
            {"comment": product_metadata[i]['comment']}
            for i in unique_indices
        ]
        
        clear_vector_db()
        
        return DeduplicationResponse(
            product=data.product,
            count=len(unique_comments),
            comments=unique_comments
        )
    
    except Exception as e:
        clear_vector_db()
        raise HTTPException(status_code=500, detail=str(e))
