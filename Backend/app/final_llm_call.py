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
    keywords: List[str]


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
