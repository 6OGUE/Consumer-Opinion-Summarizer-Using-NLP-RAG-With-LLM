import json
import requests
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

router = APIRouter()

# =======================
# MODELS
# =======================

class RedditRequest(BaseModel):
    product: str
    category: str
    aspects: List[str]
    limit: int = 100


class RedditResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    comments: List[Dict]
    sources: List[str]


HEADERS = {
    "User-Agent": "python:reddit.scraper:v1.0 (by /u/yourusername)"
}


# =======================
# MAIN FUNCTION
# =======================

@router.post("/scrape-reddit", response_model=RedditResponse)
async def scrape_reddit(request: RedditRequest):

    # =======================
    # CACHE CHECK (UPDATED)
    # =======================
    try:
        with open("reddit_data.json", "r", encoding="utf-8") as f:
            existing_data = json.load(f)

            if (
                existing_data
                and isinstance(existing_data, dict)
                and existing_data.get("product", "").lower() == request.product.lower()
                and existing_data.get("category", "") == request.category
                and existing_data.get("aspects", []) == request.aspects
                and existing_data.get("count", 0) >= request.limit
            ):
                print("Skipped scraping, Used existing data")
                return RedditResponse(**existing_data)

    except (FileNotFoundError, json.JSONDecodeError, Exception):
        pass


    # =======================
    # SEARCH THREADS
    # =======================

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
        raise HTTPException(
            status_code=404,
            detail="No threads found for the given product"
        )


    # =======================
    # SCRAPE COMMENTS
    # =======================

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

        # Main post text
        selftext = thread_json[0]["data"]["children"][0]["data"].get("selftext", "")
        if selftext.strip():
            comments.append({"comment": selftext})

        # Comments
        if len(comments) < request.limit:
            comments_raw = thread_json[1]["data"]["children"]

            for comment in comments_raw:
                if comment["kind"] == "t1":
                    body = comment["data"].get("body")

                    if body:
                        comments.append({"comment": body})

                    if len(comments) >= request.limit:
                        break

        # Track sources only if useful comments added
        if len(comments) > comments_before:
            sources.append(thread_url + "/")

        if len(comments) >= request.limit:
            break


    # =======================
    # FINAL RESPONSE (UPDATED)
    # =======================

    result = RedditResponse(
        product=request.product,
        category=request.category,   # ✅ INCLUDED
        aspects=request.aspects,     # ✅ INCLUDED
        count=len(comments),
        comments=comments,
        sources=sources,
    )


    # =======================
    # SAVE CACHE (UPDATED)
    # =======================

    with open("reddit_data.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=4, ensure_ascii=False)


    return result