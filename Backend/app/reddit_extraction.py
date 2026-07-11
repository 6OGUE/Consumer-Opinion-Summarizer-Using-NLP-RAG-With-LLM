import json
import random
import asyncio
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse, parse_qs, unquote
import sys
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright, Page, BrowserContext

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

router = APIRouter()

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


PROFILE_DIR = "./browser-profile"
MAX_THREADS = 5        
HEADLESS = True           

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


async def human_delay(min_s: float = 1.5, max_s: float = 4.0):
    """Sleep a randomized amount to avoid robotic, fixed-interval requests."""
    await asyncio.sleep(random.uniform(min_s, max_s))


def resolve_google_url(href: str) -> str:
    """Google sometimes wraps links as /url?q=<real_url>&sa=... — unwrap if needed."""
    if href.startswith("/url?"):
        params = parse_qs(urlparse(href).query)
        if "q" in params:
            return unquote(params["q"][0])
    return href

async def find_reddit_threads(page: Page, query: str, limit: int = MAX_THREADS) -> List[str]:
    search_url = f"https://www.google.com/search?q={quote_plus(query + ' reviews reddit')}&num=20"

    await page.goto(search_url, wait_until="domcontentloaded", timeout=60_000)
    await human_delay()

    # Handle occasional consent page
    try:
        await page.locator("button:has-text('Accept all')").click(timeout=3000)
        await human_delay(1, 2)
    except Exception:
        pass

    unusual_traffic = await page.locator("text=unusual traffic").count()
    if "sorry/index" in page.url or unusual_traffic > 0:
        raise HTTPException(
            status_code=503,
            detail="Google is showing a CAPTCHA / verification page. Try again later.",
        )

    await page.mouse.wheel(0, random.randint(200, 500))
    await human_delay(0.5, 1.5)

    results = page.locator("a:has(h3)")
    count = await results.count()

    thread_urls: List[str] = []
    for i in range(count):
        if len(thread_urls) >= limit:
            break

        href = await results.nth(i).get_attribute("href")
        if not href:
            continue

        url = resolve_google_url(href)
        if "reddit.com/r/" not in url or "/comments/" not in url:
            continue

        thread_url = url.split("?")[0].rstrip("/")
        if thread_url not in thread_urls:
            thread_urls.append(thread_url)

    return thread_urls

async def fetch_thread_json(page: Page, thread_url: str) -> Optional[list]:
    json_url = thread_url + ".json"

    response = await page.goto(json_url, wait_until="domcontentloaded", timeout=60_000)
    if response is None or response.status != 200:
        return None

    try:
        body_text = await page.evaluate("() => document.body.innerText")
        return json.loads(body_text)
    except (json.JSONDecodeError, Exception):
        return None




@router.post("/scrape_reddit", response_model=RedditResponse)
async def scrape_reddit(request: RedditRequest):

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

    comments: List[Dict] = []
    sources: List[str] = []

    async with async_playwright() as p:
        context: BrowserContext = await p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=HEADLESS,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            user_agent=random.choice(USER_AGENTS),
            args=["--disable-blink-features=AutomationControlled"],
        )

        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined })"
        )

        page = context.pages[0] if context.pages else await context.new_page()

        try:
            thread_urls = await find_reddit_threads(page, request.product)

            if not thread_urls:
                raise HTTPException(
                    status_code=404,
                    detail="No threads found for the given product",
                )

            for thread_url in thread_urls:
                if len(comments) >= request.limit:
                    break

                await human_delay(2, 5)  # pause between navigations, not just within a page

                thread_json = await fetch_thread_json(page, thread_url)
                if not thread_json:
                    continue

                comments_before = len(comments)
                try:
                    selftext = thread_json[0]["data"]["children"][0]["data"].get("selftext", "")
                    if selftext.strip():
                        comments.append({"comment": selftext})
                except (KeyError, IndexError, TypeError):
                    pass

                if len(comments) < request.limit:
                    try:
                        comments_raw = thread_json[1]["data"]["children"]
                    except (KeyError, IndexError, TypeError):
                        comments_raw = []

                    for comment in comments_raw:
                        if comment.get("kind") == "t1":
                            body = comment["data"].get("body")

                            if body:
                                comments.append({"comment": body})

                            if len(comments) >= request.limit:
                                break

            
                if len(comments) > comments_before:
                    sources.append(thread_url + "/")

        finally:
            await context.close()

    if not comments:
        raise HTTPException(
            status_code=404,
            detail="No comments could be scraped for the given product",
        )


    result = RedditResponse(
        product=request.product,
        category=request.category,
        aspects=request.aspects,
        count=len(comments),
        comments=comments,
        sources=sources,
    )

    with open("reddit_data.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=4, ensure_ascii=False)

    return result