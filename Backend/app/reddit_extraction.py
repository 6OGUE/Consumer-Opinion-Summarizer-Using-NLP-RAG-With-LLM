import requests
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------------- Models ----------------

class RedditRequest(BaseModel):
    product: str
    limit: int = 100

class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]
    

# ---------------- Config ----------------

PUSHSHIFT_URL = "https://api.pullpush.io/reddit/comment/search"

MIN_WORDS = 10
MIN_SCORE = 0

# ---------------- Core Logic ----------------

def fetch_comments(product: str, limit: int) -> List[Dict]:
    """
    Fetch Reddit comments using Pushshift (PullPush mirror).
    """

    params = {
        "q": product,
        "size": limit,
        "sort": "desc",
        "sort_type": "created_utc",
        "lang": "en"
    }

    try:
        response = requests.get(PUSHSHIFT_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json().get("data", [])
    except Exception as e:
        print("Pushshift error:", e)
        return []

    comments = []
    seen_ids = set()

    for item in data:
        body = item.get("body")
        if not body:
            continue

        body = body.strip()

        if body.lower() in ["[deleted]", "[removed]"]:
            continue

        if len(body.split()) < MIN_WORDS:
            continue

        score = item.get("score", 0)
        if score < MIN_SCORE:
            continue

        cid = item.get("id")
        if not cid or cid in seen_ids:
            continue

        seen_ids.add(cid)

        comments.append({
            "body": body,
        })

    return comments

# ---------------- API ----------------

@app.post("/reddit", response_model=RedditResponse)
def reddit_endpoint(request: RedditRequest):
    product = request.product.strip().lower()

    if not product:
        return RedditResponse(product=product, count=0, comments=[])

    comments = fetch_comments(product, request.limit)

    return RedditResponse(
        product=product,
        count=len(comments),
        comments=comments
    )