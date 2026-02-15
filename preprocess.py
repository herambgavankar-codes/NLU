import re

STOPWORDS = {
    "the","is","in","and","to","of","a","for","on","with","as","by","at","an","be","this","that"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return words

def preprocess_corpus(texts):
    return [clean_text(t) for t in texts]
