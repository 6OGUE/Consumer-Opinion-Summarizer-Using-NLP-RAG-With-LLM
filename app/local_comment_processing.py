from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import re
import spacy

app = FastAPI()

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model en_core_web_lg...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

nlp.max_length = 2000000

class RedditResponse(BaseModel):
    count: int
    comments: List[Dict]


URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
REDDIT_PATTERNS = [
    re.compile(r'\[deleted\]'),
    re.compile(r'\[removed\]'),
    re.compile(r'u/\w+'),
    re.compile(r'r/\w+'),
    re.compile(r'!ping\s+\w+'),
    re.compile(r'RemindMe!\s+.*'),
    re.compile(r'^\s*>\s*', re.MULTILINE),
    re.compile(r'^\s*Edit\d*:\s*', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*Update:\s*', re.MULTILINE | re.IGNORECASE),
]
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
FILLER_PHRASES = [
    'i mean', 'you know', 'like', 'basically', 'literally', 'actually',
    'tbh', 'imo', 'imho', 'fwiw', 'afaik', 'to be honest', 'in my opinion',
    'honestly', 'personally', 'i think that', 'i feel like', 'it seems like',
    'kind of', 'sort of', 'a bit', 'pretty much', 'i guess'
]

def extract_essential_meaning(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return ""
    
    doc = nlp(text)
    
    noun_phrases = []
    adjectives_with_context = []
    verbs_with_objects = []
    sentiment_words = []
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG', 'GPE', 'MONEY', 'PERCENT', 'QUANTITY']:
            entities.append(ent.text)
    
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if len(chunk_text) > 2:
            noun_phrases.append(chunk_text)
    
    for token in doc:
        if token.pos_ == 'ADJ':
            head_text = token.head.text if token.head.pos_ in ['NOUN', 'PROPN'] else ""
            if head_text:
                adjectives_with_context.append(f"{token.text} {head_text}")
            else:
                sentiment_words.append(token.text)
        
        elif token.pos_ == 'VERB' and not token.is_stop:
            obj = [child.text for child in token.children if child.dep_ in ['dobj', 'attr', 'prep']]
            if obj:
                verbs_with_objects.append(f"{token.text} {' '.join(obj[:2])}")
            else:
                verbs_with_objects.append(token.text)
        
        elif token.dep_ == 'neg':
            negation_phrase = f"not {token.head.text}"
            sentiment_words.append(negation_phrase)
    
    result_parts = []

    result_parts.extend(entities[:3])
    
    unique_noun_phrases = []
    seen = set()
    for phrase in noun_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower not in seen and not any(phrase_lower in e.lower() for e in entities):
            unique_noun_phrases.append(phrase)
            seen.add(phrase_lower)

    result_parts.extend(unique_noun_phrases[:5])
    result_parts.extend(adjectives_with_context[:4])
    result_parts.extend(sentiment_words[:3])
    result_parts.extend(verbs_with_objects[:3])
    
    return ' '.join(result_parts)

def clean_comment(text: str, comment_key: str = "body") -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = URL_PATTERN.sub('', text)

    for pattern in REDDIT_PATTERNS:
        text = pattern.sub('', text)
    
    text = EMOJI_PATTERN.sub('', text)
    
    for phrase in FILLER_PHRASES:
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) > 20:
        text = extract_essential_meaning(text)
    
    return text.strip()

@app.post("/process_comments")
async def clean_comments(reddit_data: RedditResponse, comment_key: str = "body"):
    cleaned_comments = {}
    
    for idx, comment in enumerate(reddit_data.comments):
        original_text = comment.get(comment_key, "") if isinstance(comment, dict) else str(comment)
        cleaned_text = clean_comment(original_text, comment_key=comment_key)
        cleaned_comments[idx] = cleaned_text
    
    return {"comments": cleaned_comments}