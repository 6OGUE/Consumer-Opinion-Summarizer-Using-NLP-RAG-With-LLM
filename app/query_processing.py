import warnings
import torch
from gliner import GLiNER
from typing import Optional, Tuple

#################### This is for cleaner terminal output #########################
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
###################################################################################

PRODUCT_LABELS = ["product", "brand", "item", "model", "device"]
gliner_model: Optional[GLiNER] = None # prevent unnecessary loading

def _initialize_gliner_model(model_name="urchade/gliner_medium-v2.1"): 
    global gliner_model
    if gliner_model is not None:
        return 

    try:
        gliner_model = GLiNER.from_pretrained(model_name, local_files_only=True) # check if local copy avaailable
    except Exception:
        print(f"Downloading GLiNER model ('{model_name}' - first time only)...") # only download if no local copy
        gliner_model = GLiNER.from_pretrained(model_name)

_initialize_gliner_model()


def extract_product_name(query: str, threshold: float = 0.4):
    if not query or not gliner_model:
        return None
    normalized_query = query.strip()
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
            entity_label = entity["label"].lower()
            if entity_label in [label.lower() for label in PRODUCT_LABELS]:
                product_parts.append(entity["text"])
        
        if product_parts:
            return max(product_parts, key=len)
        else:
            return None

    except Exception as e:
        print(f"Product extraction failed: {e}")
        return None




def validate_and_extract_product(query: str, threshold: float = 0.4) -> Tuple[bool, str]:
    if not query or not query.strip():
        result=False
        emptiness=True
        suggestions=None
        extracted=None
        return result, emptiness, suggestions, extracted
    
    product = extract_product_name(query, threshold)
    
    if product is None:
        result=False
        emptiness=False
        suggestions=None
        extracted=None
        return result, emptiness, suggestions, extracted
    extracted_lower = product.strip().lower()
    query_lower = query.strip().lower()
    if extracted_lower == query_lower:
        result=True
        emptiness=False
        suggestions=None
        extracted=query
        return result, emptiness, suggestions, extracted
    if extracted_lower in query_lower:
        result=False
        emptiness=False
        suggestions=product.strip()
        extracted=None
        return result, emptiness, suggestions, extracted
    return False, f"No Match || Suggestions: {product.strip()}"



print(validate_and_extract_product("Asus Tuff Gaming Laptop"),end='')
print(" // result, emptiness, suggestions, extracted")
