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

class CommentResult(BaseModel):
    summary: str
    sentiment: str          


class ProcessingResponse(BaseModel):
    product: str
    count: int
    comments: Dict[int, CommentResult]


class FinalInsightResponse(BaseModel):
    overview: str
    unique_features: List[str]
    strengths: List[str]
    weaknesses: List[str]
    alternatives: List[str]
    final_insight: str


def final_llm_call(data: ProcessingResponse) -> FinalInsightResponse:

    if os.getenv("test"):
        return FinalInsightResponse(
            overview="iPhone 14 is Apple’s 2022 mainstream flagship phone, featuring a 6.1-inch Super Retina XDR OLED display, A15 Bionic chip, dual-camera system, and iOS 16, still supported in 2026.",
    unique_features=[
        "Emergency SOS via satellite",
        "Crash Detection automatically calls emergency services",
        "Photonic Engine for better low-light photos"
    ],
    strengths=[
        "Reliable, smooth everyday performance",
        "Solid battery life lasting a full day",
        "Consistently excellent camera output for casual photography",
        "Stable and optimized iOS experience"
    ],
    weaknesses=[
        "No 120 Hz display",
        "Older A15 chip compared to Pro models",
        "Uses Lightning port instead of USB-C",
        "Limited camera versatility with no telephoto lens"
    ],
    alternatives=[
        "iPhone 14 Pro with faster chip, 120 Hz display, advanced cameras",
        "OnePlus 11",
        "Xiaomi 13 Pro",
        "Sony Xperia 1 IV for 4K screen and advanced camera controls"
    ],
    final_insight="iPhone 14 is a dependable and refined device, still fully usable in 2026, ideal for smooth software and consistent camera performance, but lacks cutting-edge hardware, so tech enthusiasts might prefer alternatives."
        )
    
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
    - The overview and final_insight must be 2–3 sentences, coherent and well-structured

    Return ONLY valid JSON matching this exact schema:
    {{
        "overview": "string - a formal 2-3 sentence product summary",
        "unique_features": ["string array - formally stated unique product features"],
        "strengths": ["string array - professionally rephrased positive aspects derived from user comments"],
        "weaknesses": ["string array - professionally rephrased negative aspects derived from user comments"],
        "alternatives": ["string array - alternative product names mentioned"],
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
        })
    result_dict = json.loads(response.text)
    validated = FinalInsightResponse(**result_dict)
    return validated

@router.post("/llm", response_model=FinalInsightResponse)
def llm_process(data: ProcessingResponse):
    
    try:
        result = final_llm_call(data)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid LLM response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")
