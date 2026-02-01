import warnings
from gliner import GLiNER
from typing import Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

#################### Cleaner terminal output #########################
import os
import logging
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="transformers.convert_slow_tokenizer"
)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
########################################################################

PRODUCT_DICT: set[str] = set()

def load_product_dictionary(file_path="products.txt"):
    global PRODUCT_DICT
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            
            PRODUCT_DICT = set(line.strip().lower() for line in f if line.strip())
        print(f"Loaded {len(PRODUCT_DICT)} products from {file_path}")
    except FileNotFoundError:
        print(f"Product dictionary file '{file_path}' not found. Continuing without dictionary.")

load_product_dictionary()

PRODUCT_LABELS = ["product", "brand", "item", "model", "device"]
gliner_model: Optional[GLiNER] = None

def _initialize_gliner_model(model_name="urchade/gliner_large-v2.1"): 
    global gliner_model
    if gliner_model is not None:
        return 

    try:
        gliner_model = GLiNER.from_pretrained(model_name, local_files_only=True)
    except Exception:
        print(f"Downloading GLiNER model ('{model_name}' - first time only)...")
        gliner_model = GLiNER.from_pretrained(model_name)

_initialize_gliner_model()

def extract_product_name(query: str, threshold: float = 0.4):
    if not query:
        return None

    normalized_query = query.strip().lower()
    for product in PRODUCT_DICT:
        if product in normalized_query or product.startswith(normalized_query):
            return product     

 
    if not gliner_model:
        return None

    try:
        entities = gliner_model.predict_entities(
            normalized_query, 
            PRODUCT_LABELS, 
            threshold=threshold,
            flat_ner=True  
        )
        
        if not entities:
            return None
        
        product_parts = []
        for entity in entities:
            if entity["label"].lower() in [label.lower() for label in PRODUCT_LABELS]:
                product_parts.append(entity["text"])
        
        if product_parts:
            return max(product_parts, key=len)
        return None

    except Exception as e:
        print(f"Product extraction failed: {e}")
        return None

def validate_and_extract_product(query: str, threshold: float = 0.4) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    if not query or not query.strip():
        return False, True, None, None
    
    product = extract_product_name(query, threshold)
    
    if product is None:
        return False, False, None, None
        
    extracted_lower = product.strip().lower()
    query_lower = query.strip().lower()
    
    if extracted_lower == query_lower:
        return True, False, None, query
        
    if extracted_lower in query_lower:
        return False, False, product.strip(), None
        
    return False, False, product.strip(), None

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    threshold: float = 0.4

class QueryResponse(BaseModel):
    result: bool
    emptiness: bool
    suggestion: Optional[str]
    extracted: Optional[str]

@app.post("/local", response_model=QueryResponse)
def local_process(request: QueryRequest):
    query = request.query.strip().lower()
    result, emptiness, suggestion, extracted = validate_and_extract_product(query, request.threshold)
    
    return QueryResponse(
        result=result,
        emptiness=emptiness,
        suggestion=suggestion,
        extracted=extracted
    )
