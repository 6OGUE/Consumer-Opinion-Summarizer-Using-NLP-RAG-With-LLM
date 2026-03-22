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
    You are an opinion summarization system.

    Task: Analyze the following ProcessingResponse JSON and create summarized insights.

    Return ONLY valid JSON matching this exact schema:
    {{
    "overview": "string - brief product summary",
    "unique_features": ["string array - unique product features"],
    "strengths": ["string array - positive aspects - from the comments only"],
    "weaknesses": ["string array - negative aspects - from the comments only"],  
    "alternatives": ["string array - alternative product names"],
    "final_insight": "string - conclusion based on the analysis"
    }}

    CRITICAL: Base all points strictly on the provided comments, keywords and sentiments. No assumptions.

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
