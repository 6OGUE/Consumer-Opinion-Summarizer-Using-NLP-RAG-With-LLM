import os
from google import genai
import json
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv
from fastapi import APIRouter

router = APIRouter()

load_dotenv()
api_key = os.getenv("api_key")
client = genai.Client(api_key=api_key)

CACHE_FILE = "llm_data.json"


# ✅ CHANGED: sentiment → sentiments
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


class FinalInsightResponse(BaseModel):
    product: str
    category: str
    aspects: List[str]
    count: int
    overview: str
    unique_features: List[str]
    strengths: List[str]
    weaknesses: List[str]
    alternatives: List[str]
    final_insight: str


def load_cached_response(product: str, count: int, category: str, aspects: List[str]) -> Optional[FinalInsightResponse]:
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, "r") as f:
            cached_data = json.load(f)
        
        if (
            cached_data.get("product") == product and
            cached_data.get("count") == count and
            cached_data.get("category") == category and
            cached_data.get("aspects") == aspects
        ):
            return FinalInsightResponse(**cached_data)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    return None


def save_response_to_cache(response: FinalInsightResponse) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(response.model_dump(), f, indent=2)


def final_llm_call(data: ProcessingResponse) -> FinalInsightResponse:
    input_json = data.model_dump_json(indent=2)

    prompt = f"""
    You are an expert product analyst and technical writer specializing in consumer opinion synthesis.

    Task: Analyze the following ProcessingResponse JSON and generate a structured, professional product intelligence report.

    TONE & STYLE REQUIREMENTS:
    - All output must be written in formal, professional English
    - Rephrase raw or colloquial user comments into polished, analytical statements
    - Avoid casual language, slang, or direct quote fragments
    - Express insights as objective observations, not personal opinions
    - Use precise, industry-appropriate vocabulary

    FORMATTING RULES:
    - Each array item must be a complete, standalone sentence
    - Sentences must follow Subject → Predicate → Object structure where applicable
    - Begin strength/weakness points with action-oriented or descriptive phrases
      (e.g., "Demonstrates superior...", "Exhibits a notable lack of...", "Users consistently report...")
    - The overview and final_insight must be 2-3 sentences, coherent and well-structured

    Return ONLY valid JSON matching this exact schema:
    {{
        "overview": "string - a formal 2-3 sentence product summary",
        "unique_features": ["string array - formally stated highlighted features derived from user comments"],
        "strengths": ["string array - professionally rephrased positive aspects derived from user comments"],
        "weaknesses": ["string array - professionally rephrased negative aspects derived from user comments"],
        "alternatives": ["string array - alternative product names mentioned(maximum 3)"],
        "final_insight": "string - a formal 2-3 sentence conclusion grounded in the analysis"
    }}

    TRANSFORMATION EXAMPLES (for tone calibration):
    - Raw: "i love my iphone"       → Formal: "Users express strong overall satisfaction with the device."
    - Raw: "battery dies too fast"  → Formal: "Battery longevity is frequently cited as a significant area of concern."
    - Raw: "camera is insane"       → Formal: "The camera system consistently receives exceptional praise for its performance."
    - Raw: "way too expensive"      → Formal: "The product's pricing is perceived as a barrier to accessibility by a notable portion of users."

    CRITICAL CONSTRAINTS:
    - Base all insights strictly on the provided comments, keywords, and sentiments — no assumptions or external knowledge
    - Do not fabricate features, strengths, weaknesses, or alternatives not present in the data
    - Do not include raw user quotes or informal phrasing in the output

    Input Data:
    {input_json}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 1000,
            "response_mime_type": "application/json"
        }
    )

    result_dict = json.loads(response.text)

    # ✅ propagate unchanged
    result_dict["product"] = data.product
    result_dict["category"] = data.category
    result_dict["aspects"] = data.aspects
    result_dict["count"] = data.count

    validated = FinalInsightResponse(**result_dict)
    return validated


@router.post("/llm", response_model=FinalInsightResponse)
def llm_process(data: ProcessingResponse):
    try:
        cached = load_cached_response(
            data.product,
            data.count,
            data.category,
            data.aspects
        )
        if cached:
            return cached

        result = final_llm_call(data)
        save_response_to_cache(result)
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid LLM response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")