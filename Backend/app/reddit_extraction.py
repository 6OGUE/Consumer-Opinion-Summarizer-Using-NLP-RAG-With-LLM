import asyncio
import json
import random
import sys
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from playwright.async_api import (
    BrowserContext,
    Page,
    async_playwright,
)

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
HEADLESS = False

USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
]


async def human_delay(min_seconds: float = 1.5, max_seconds: float = 4.0):
    await asyncio.sleep(random.uniform(min_seconds, max_seconds))


def resolve_google_url(href: str) -> Optional[str]:
    if not href.startswith("/url?"):
        return None

    parsed = urlparse(href)
    params = parse_qs(parsed.query)

    if "q" in params and params["q"]:
        return unquote(params["q"][0])

    return None


def is_reddit_thread(url: str) -> bool:
    if not url:
        return False

    url_lower = url.lower()
    return "reddit.com/r/" in url_lower and "/comments/" in url_lower


def clean_reddit_url(url: str) -> str:
    return url.split("?")[0].split("#")[0]


async def find_reddit_threads(page: Page, product: str, category: str) -> List[str]:
    query = f"{product} {category} reviews reddit"
    google_url = "https://www.google.com/search?q=" + quote_plus(query)

    print(f"\nSearching Google for: {query}")

    await page.goto(google_url, wait_until="domcontentloaded", timeout=60000)
    await human_delay(2, 4)

    try:
        accept_button = page.locator("button:has-text('Accept all')")
        if await accept_button.count() > 0:
            await accept_button.first.click(timeout=3000)
            await human_delay(1, 2)
    except Exception:
        pass

    unusual_traffic = await page.locator("text=unusual traffic").count()
    if "sorry/index" in page.url or unusual_traffic > 0:
        raise HTTPException(
            status_code=503,
            detail="Google is showing a CAPTCHA / verification page. Try again later.",
        )

    links = await page.locator("a").all()
    print(f"Google result links found: {len(links)}")

    reddit_threads: List[str] = []
    seen_hrefs = set()

    for link in links:
        if len(reddit_threads) >= MAX_THREADS:
            break

        try:
            href = await link.get_attribute("href")
            if not href:
                continue

            if href.startswith("/url?"):
                resolved_url = resolve_google_url(href)
                if not resolved_url:
                    continue

                if is_reddit_thread(resolved_url):
                    clean_url = clean_reddit_url(resolved_url)
                    if clean_url not in reddit_threads:
                        reddit_threads.append(clean_url)
                        print(f"Found Reddit thread: {clean_url}")

            elif href.startswith("/goto?"):
                if href in seen_hrefs:
                    continue
                seen_hrefs.add(href)

                aria_label = await link.get_attribute("aria-label") or ""
                if "reddit" not in aria_label.lower():
                    continue

                if not await link.is_visible():
                    continue

                try:
                    async with page.context.expect_page(timeout=8000) as new_page_info:
                        await link.click(force=True, timeout=5000)

                    new_page = await new_page_info.value

                    try:
                        await new_page.wait_for_load_state("domcontentloaded", timeout=15000)
                    except Exception:
                        pass

                    final_url = new_page.url

                    if is_reddit_thread(final_url):
                        clean_url = clean_reddit_url(final_url)
                        if clean_url not in reddit_threads:
                            reddit_threads.append(clean_url)
                            print(f"Found Reddit thread: {clean_url}")

                    await new_page.close()

                except Exception as e:
                    print(f"Could not resolve Google /goto result: {e}")

        except Exception as e:
            print(f"Error processing Google link: {e}")

    unique_threads: List[str] = []
    for thread in reddit_threads:
        if thread not in unique_threads:
            unique_threads.append(thread)

    print(f"\nTotal Reddit threads found: {len(unique_threads)}")
    return unique_threads[:MAX_THREADS]


async def fetch_thread_json(page: Page, thread_url: str) -> Optional[list]:
    json_url = thread_url.rstrip("/") + ".json"

    print("\nFetching:")
    print(json_url)

    try:
        response = await page.goto(json_url, wait_until="domcontentloaded", timeout=60000)

        if response is None or response.status != 200:
            print(f"HTTP status: {response.status if response else 'no response'}")
            return None

        print(f"HTTP status: {response.status}")

        await human_delay(1, 2)

        body_text = await page.locator("body").inner_text()

        if not body_text.strip():
            print("Reddit returned an empty response.")
            return None

        return json.loads(body_text)

    except json.JSONDecodeError as e:
        print(f"Failed to decode Reddit JSON: {e}")
        return None

    except Exception as e:
        print(f"Failed to fetch Reddit JSON: {e}")
        return None


@router.post("/scrape_reddit", response_model=RedditResponse)
async def scrape_reddit(request: RedditRequest):
    if request.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be greater than 0")

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
            print("Using existing reddit_data.json")
            return RedditResponse(**existing_data)

    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        pass

    comments: List[Dict] = []
    sources: List[str] = []

    async with async_playwright() as p:
        user_agent = random.choice(USER_AGENTS)

        context: BrowserContext = await p.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=HEADLESS,
            user_agent=user_agent,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )

        await context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """
        )

        page = context.pages[0] if context.pages else await context.new_page()

        try:
            threads = await find_reddit_threads(page, request.product, request.category)

            if not threads:
                raise HTTPException(
                    status_code=404,
                    detail="No threads found for the given product",
                )

            for index, thread_url in enumerate(threads):
                if len(comments) >= request.limit:
                    break

                print(f"\n--- Thread {index + 1}/{len(threads)} ---")

                await human_delay(2, 5)

                thread_json = await fetch_thread_json(page, thread_url)
                if not thread_json:
                    continue

                try:
                    if not isinstance(thread_json, list):
                        print("Unexpected Reddit JSON structure.")
                        continue

                    if len(thread_json) < 2:
                        print("Reddit JSON does not contain comments.")
                        continue

                    post_data = thread_json[0].get("data", {}).get("children", [])

                    if post_data:
                        post = post_data[0].get("data", {})
                        selftext = post.get("selftext")
                        if selftext:
                            comments.append({"comment": selftext})

                    comment_children = thread_json[1].get("data", {}).get("children", [])
                    extracted_from_thread = False

                    for child in comment_children:
                        if len(comments) >= request.limit:
                            break

                        if child.get("kind") != "t1":
                            continue

                        comment_data = child.get("data", {})
                        body = comment_data.get("body")

                        if not body:
                            continue

                        comments.append({"comment": body})
                        extracted_from_thread = True

                    if extracted_from_thread or post_data:
                        if thread_url not in sources:
                            sources.append(thread_url)

                    print(f"Extracted {len(comments)} comments")

                except Exception as e:
                    print(f"Error extracting thread: {e}")

        finally:
            await context.close()

    if not comments:
        raise HTTPException(status_code=404, detail="No Reddit comments found")

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
