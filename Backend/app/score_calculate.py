from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

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

def calculate_scores(comments: ProcessingResponse) -> float:
    score = 0
    total = 0

    for comment in comments.comments.values():
        total += 1

        if comment.sentiment == "positive":
            score += 1
        elif comment.sentiment == "negative":
            score -= 1

    if total == 0:
        return 0

    score = ((score + total) / (total * 2)) * 100
    return score

app = FastAPI()

@app.post("/calc_score", response_model=ScoreResponse)
def calc_score(data: ProcessingResponse):
    score = calculate_scores(data)
    return ScoreResponse(score=score)
