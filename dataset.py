from utils.logger import log

def load_dataset(path):
    log(f"Reading dataset from {path}")
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) != 2:
                continue
            category, text = parts
            if category not in ["sport", "politics"]:
                continue
            texts.append(text)
            labels.append(0 if category == "sport" else 1)

    log(f"Loaded {len(texts)} samples")
    return texts, labels
