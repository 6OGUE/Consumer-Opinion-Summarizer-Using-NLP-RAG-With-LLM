import json
from app.deduplicate_comments import deduplicate_comments
from app.capture_semantics import process_data
from app.local_comment_processing import clean_comments
from fastapi import FastAPI
from app.capture_semantics import process_data, DeduplicationResponse
from app.capture_semantics import RedditResponse
app=FastAPI()

with open("test_comments.json", "r") as f:
    raw_data = json.load(f)
data = DeduplicationResponse(**raw_data)

@app.post("/pipeline")
async def run_pipeline():
    #Skipped product name extractions(local, llm)
    #Skipped Reddit extraction and used a dummy(test_comments.json)
    dedu=await deduplicate_comments(data)
    if isinstance(dedu, dict):
        dedu = DeduplicationResponse(**dedu)
    capturesem=process_data(dedu)
    if isinstance(capturesem, dict):
        capturesem = RedditResponse(**capturesem)
    clean=await clean_comments(capturesem)
    return clean
    #Skipped final llm call and manually called a hosted llm using the "clean" => ✅ Works
    #Skipped score calculation and manually passed "clean" to the score calculator => ✅ Works

    