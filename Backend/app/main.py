from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.local_query_processing import router as product_extract_local
from app.llm_query_proessing_hosted import router as product_extract_hosted
from app.reddit_extraction import router as reddit_extract
from app.deduplicate_comments import router as remove_duplicates
from app.capture_semantics import router as cleanup_comments
from app.local_comment_processing import router as process_comments
from app.score_calculate import router as score_finder
from app.final_llm_call import router as final_call
from app.chatbot import router as chatbot
from app.classify_product import router as classify_product

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 1
app.include_router(product_extract_local, prefix="/product_extract_local", tags=["Product Extract Local"]) 
# Step 1 Fallback
app.include_router(product_extract_hosted, prefix="/product_extract_hosted", tags=["Product Extract Hosted"]) 
# Step 2
app.include_router(classify_product, prefix="/classify_product", tags=["Classify Product"]) 
# Step 3
app.include_router(reddit_extract, prefix="/reddit_extract", tags=["Reddit Extract"])
# Step 4
app.include_router(remove_duplicates, prefix="/remove_duplicates", tags=["Remove Duplicates"])
# Step 5
app.include_router(cleanup_comments, prefix="/cleanup_comments", tags=["Cleanup Comments"])
# Step 6
app.include_router(chatbot, prefix="/chatbot", tags=["Chatbot"])
# Step 7
app.include_router(process_comments, prefix="/process_comments", tags=["Process Comments"])
# Step 8
app.include_router(score_finder, prefix="/score_finder", tags=["Score Finder"])
# Step 9
app.include_router(final_call, prefix="/final_call", tags=["Final Call"])
