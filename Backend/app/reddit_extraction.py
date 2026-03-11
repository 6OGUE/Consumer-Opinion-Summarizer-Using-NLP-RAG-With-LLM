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
    sources: List[str]

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
        "limit": 3,
    }
    search_response = requests.get(search_url, headers=HEADERS, params=params)
    if search_response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Reddit search failed with status {search_response.status_code}",
        )

    search_data = search_response.json()
    all_threads = search_data["data"]["children"]

    qualifying_threads = [
        t for t in all_threads
        if request.product.lower() in t["data"]["title"].lower()
    ]

    if not qualifying_threads:
        raise HTTPException(status_code=404, detail="No threads found for the given product")

    comments = []
    sources = []

    for thread in qualifying_threads:
        thread_data = thread["data"]
        thread_url = "https://old.reddit.com" + thread_data["permalink"].rstrip("/")

        thread_json_url = thread_url + ".json"

        thread_response = requests.get(thread_json_url, headers=HEADERS)
        if thread_response.status_code != 200:
            continue

        thread_json = thread_response.json()

        comments_before = len(comments)

        selftext = thread_json[0]["data"]["children"][0]["data"].get("selftext", "")
        if selftext.strip():
            comments.append({"comment": selftext})

        if len(comments) < request.limit:
            comments_raw = thread_json[1]["data"]["children"]
            for comment in comments_raw:
                if comment["kind"] == "t1":
                    comments.append({"comment": comment["data"].get("body")})
                    if len(comments) >= request.limit:
                        break

        if len(comments) > comments_before:
             sources.append(thread_url + "/")

        if len(comments) >= request.limit:
            break

    result = RedditResponse(
        product=request.product,
        count=len(comments),
        comments=comments,
        sources=sources,
    )

    with open("reddit_data.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=4, ensure_ascii=False)

    return result