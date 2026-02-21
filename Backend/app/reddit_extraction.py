import json
import requests
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class RedditRequest(BaseModel):
    product: str
    limit: int = 100

class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]

HEADERS = {
    "User-Agent": "python:reddit.scraper:v1.0 (by /u/yourusername)"
}

@router.post("/scrape-reddit", response_model=RedditResponse)
def scrape_reddit(request: RedditRequest):

    search_url = "https://old.reddit.com/search.json"
    params = {
        "q": f"{request.product} review",
        "type": "link",
        "sort": "relevance",
        "limit": 5,
    }
    search_response = requests.get(search_url, headers=HEADERS, params=params)
    if search_response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Reddit search failed with status {search_response.status_code}",
        )

    search_data = search_response.json()

    threads = search_data["data"]["children"]

    if not threads:
        raise HTTPException(status_code=404, detail="No threads found for the given product")

    thread = threads[0]["data"]
    thread_url = "https://old.reddit.com" + thread["permalink"]

    thread_json_url = thread_url.rstrip("/") + ".json"
    thread_response = requests.get(thread_json_url, headers=HEADERS)
    if thread_response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch thread data from Reddit")

    thread_data = thread_response.json()

    comments_raw = thread_data[1]["data"]["children"]

    comments = []
    for comment in comments_raw:
        if comment["kind"] == "t1":
            comment_data = comment["data"]
            comments.append(
                {
                    "comment": comment_data.get("body"),
                }
            )
            if len(comments) >= request.limit:
                break

    result = RedditResponse(
        product=request.product,
        count=len(comments),
        comments=comments,
    )

    with open("reddit_data.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=4, ensure_ascii=False)

    return result