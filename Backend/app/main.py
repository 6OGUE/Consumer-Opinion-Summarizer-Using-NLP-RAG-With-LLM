from fastapi import FastAPI
from app.reddit_extraction import router as reddit_router
from app.local_query_processing import router as local_query_router
from app.local_comment_processing import router as local_comment_router
from app.llm_query_processing import router as llm_query_router
from app.final_llm_call import router as final_llm_router
from app.score_calculate import router as score_router
from app.deduplicate_comments import router as deduplicate_router

app = FastAPI()

app.include_router(reddit_router, prefix="/reddit")
app.include_router(local_query_router, prefix="/query")
app.include_router(local_comment_router, prefix="/comment")
app.include_router(llm_query_router, prefix="/llm-query")
app.include_router(final_llm_router, prefix="/llm-call")
app.include_router(score_router, prefix="/score")
app.include_router(deduplicate_router, prefix="/deduplicate")