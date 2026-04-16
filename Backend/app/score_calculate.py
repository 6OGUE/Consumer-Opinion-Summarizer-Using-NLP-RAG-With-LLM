from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()


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


class ScoreResponse(BaseModel):
    overall_score: float
    aspect_scores: Dict[str, float]


def sentiment_to_score(sentiment: str) -> int:
    if sentiment == "positive":
        return 1
    elif sentiment == "negative":
        return -1
    return 0


def normalize(score: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(((score + total) / (2 * total)) * 100)


def calculate_scores(data: ProcessingResponse):

    # 🔹 overall sentiment aggregation
    overall_score = 0
    overall_total = 0

    # 🔹 aspect-wise aggregation
    aspect_scores = {aspect: 0 for aspect in data.aspects}
    aspect_totals = {aspect: 0 for aspect in data.aspects}

    for comment in data.comments.values():

        # ✅ overall sentiment
        overall_score += sentiment_to_score(comment.overall_sentiment)
        overall_total += 1

        # ✅ aspect sentiments
        for aspect in data.aspects:
            sentiment = comment.sentiments.get(aspect, "neutral")

            aspect_scores[aspect] += sentiment_to_score(sentiment)
            aspect_totals[aspect] += 1

    # 🔹 normalize
    overall_final = normalize(overall_score, overall_total)

    aspect_final = {
        aspect: normalize(aspect_scores[aspect], aspect_totals[aspect])
        for aspect in data.aspects
    }

    return overall_final, aspect_final


@router.post("/calc_score", response_model=ScoreResponse)
def calc_score(data: ProcessingResponse):
    overall, aspects = calculate_scores(data)

    return ScoreResponse(
        overall_score=overall,
        aspect_scores=aspects
    )