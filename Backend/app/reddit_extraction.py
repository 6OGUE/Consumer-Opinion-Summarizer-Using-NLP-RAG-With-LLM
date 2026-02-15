import requests
import time
from typing import List, Dict
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

class RedditRequest(BaseModel):
    product: str
    limit: int = 100

class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]

SUBMISSION_SEARCH_URL = "https://api.pullpush.io/reddit/submission/search"
COMMENT_SEARCH_URL = "https://api.pullpush.io/reddit/comment/search"

def fetch_comments(product: str, limit: int) -> List[Dict]:
    all_comments = []
    seen_bodies = set()
    
    try:
        thread_res = requests.get(SUBMISSION_SEARCH_URL, params={
            "q": f"{product} review",
            "size": 5,
            "sort": "desc",
            "sort_type": "score"
        }, timeout=10)
        thread_ids = [f"t3_{item['id']}" for item in thread_res.json().get("data", []) if "id" in item]
    except:
        thread_ids = []

    for tid in thread_ids:
        try:
            res = requests.get(COMMENT_SEARCH_URL, params={"link_id": tid, "size": 40}, timeout=10)
            if res.status_code == 200:
                for item in res.json().get("data", []):
                    body = item.get("body", "").strip()
                    
                    if (len(body.split()) > 10 and 
                        body not in seen_bodies and 
                        "fakespot" not in body.lower()):
                        all_comments.append({"body": body})
                        seen_bodies.add(body)
            time.sleep(0.1) 
        except:
            continue

    if len(all_comments) < 10:
        try:
            res = requests.get(COMMENT_SEARCH_URL, params={
                "q": f'"{product}" review',
                "size": limit,
                "sort": "desc"
            }, timeout=10)
            if res.status_code == 200:
                for item in res.json().get("data", []):
                    body = item.get("body", "").strip()
                    if (len(body.split()) > 15 and 
                        body not in seen_bodies and 
                        "fakespot" not in body.lower()):
                        all_comments.append({"body": body})
                        seen_bodies.add(body)
        except:
            pass

    return all_comments[:limit]

@router.post("/reddit", response_model=RedditResponse)
def reddit_endpoint(request: RedditRequest):
    product_query = request.product.strip()
    
    if not product_query or product_query.lower() == "string":
        return RedditResponse(product=product_query, count=0, comments=[])

    comments_list = fetch_comments(product_query, request.limit)
    
    return RedditResponse(
        product=product_query,
        count=len(comments_list),
        comments=comments_list
    )