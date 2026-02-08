import requests
import re
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict

app = FastAPI()

# =====================================================
# Models
# =====================================================

class RedditRequest(BaseModel):
    product: str                 # normalized product name (e.g. "iphone")
    max_discussions: int = 5
    comments_per_discussion: int = 5

class RedditResponse(BaseModel):
    product: str
    count: int
    comments: List[Dict]

# =====================================================
# Config
# =====================================================

COMMENT_SEARCH_URL = "https://api.pullpush.io/reddit/comment/search"

MIN_COMMENT_WORDS = 10
FETCH_SIZE = 100   # how many comments to fetch initially

# =====================================================
# Cleaning
# =====================================================

EMOJI_PATTERN = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)  # remove quotes
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# Core Logic (DISCUSSION-FIRST, RELIABLE)
# =====================================================

def fetch_comments_containing_product(product: str) -> List[Dict]:
    """
    Step 1: Fetch comments that explicitly mention the product.
    This is the ONLY reliable Pushshift search.
    """
    params = {
        "q": product,
        "size": FETCH_SIZE,
        "sort": "desc",
        "sort_type": "score"
    }

    try:
        r = requests.get(COMMENT_SEARCH_URL, params=params, timeout=20)
        return r.json().get("data", [])
    except Exception:
        return []

def group_by_discussion(comments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group comments by Reddit discussion (link_id).
    Each link_id corresponds to one discussion thread.
    """
    grouped = defaultdict(list)

    for c in comments:
        link_id = c.get("link_id")
        body = c.get("body", "")

        if not link_id:
            continue

        if len(body.split()) < MIN_COMMENT_WORDS:
            continue

        grouped[link_id].append(c)

    return grouped

def extract_discussion_comments(
    product: str,
    max_discussions: int,
    comments_per_discussion: int
) -> List[Dict]:
    """
    Final logic:
    1. Discover discussions via comment search
    2. Group by discussion (link_id)
    3. Extract top comments per discussion
    """
    raw_comments = fetch_comments_containing_product(product)
    grouped = group_by_discussion(raw_comments)

    results = []

    # Sort discussions by number of product-mentioning comments
    sorted_discussions = sorted(
        grouped.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for link_id, comments in sorted_discussions[:max_discussions]:
        # Take top comments from this discussion
        for c in comments[:comments_per_discussion]:
            cleaned = clean_text(c.get("body", ""))

            if len(cleaned.split()) < MIN_COMMENT_WORDS:
                continue

            results.append({
                "discussion_id": link_id,
                "comment": cleaned,
                "subreddit": c.get("subreddit"),
                "score": c.get("score", 0)
            })

    return results

# =====================================================
# API
# =====================================================

@app.post("/reddit", response_model=RedditResponse)
def reddit_endpoint(request: RedditRequest):
    product = request.product.strip().lower()

    if not product:
        return RedditResponse(product=product, count=0, comments=[])

    comments = extract_discussion_comments(
        product,
        request.max_discussions,
        request.comments_per_discussion
    )

    return RedditResponse(
        product=product,
        count=len(comments),
        comments=comments
    )
