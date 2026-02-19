from fastapi import FastAPI,APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class CommentResult(BaseModel):
    summary: str
    sentiment: str
    keywords: List[str]


class ProcessingResponse(BaseModel):
    product: str
    count: int
    comments: Dict[int, CommentResult]


class ScoreResponse(BaseModel):
    score: float


def calculate_scores(data: ProcessingResponse) -> float:
    score = 0
    total = len(data.comments)

    for comment in data.comments.values():
        if comment.sentiment == "positive":
            score += 1
        elif comment.sentiment == "negative":
            score -= 1

    if total == 0:
        return 0.0

    return ((score + total) / (total * 2)) * 100

@router.post("/calc_score", response_model=ScoreResponse)
def calc_score(data: ProcessingResponse):
    score = calculate_scores(data)
    return ScoreResponse(score=score)
