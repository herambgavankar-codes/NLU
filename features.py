from collections import Counter, defaultdict
import math
from utils.logger import log


# ---------------------------------------------------------
# 1️⃣ BUILD VOCAB (common for both BoW and TFIDF)
# ---------------------------------------------------------

def build_vocab(corpus, max_features=3000):
    log("Counting word frequencies for vocabulary")

    freq = Counter()
    for doc in corpus:
        freq.update(doc)

    most_common = freq.most_common(max_features)

    vocab = {word: i for i, (word, _) in enumerate(most_common)}

    log(f"Vocabulary created with size = {len(vocab)}")
    return vocab


# ---------------------------------------------------------
# 2️⃣ BAG OF WORDS (SPARSE)
# ---------------------------------------------------------

def bow_vector_sparse(doc, vocab):
    counts = Counter(doc)
    vec = {}

    for word, c in counts.items():
        if word in vocab:
            vec[vocab[word]] = c

    return vec


def build_bow_features(corpus, vocab):
    log("Building Bag-of-Words sparse vectors")
    vectors = [bow_vector_sparse(doc, vocab) for doc in corpus]
    log("BoW feature construction complete")
    return vectors


# ---------------------------------------------------------
# 3️⃣ TF-IDF (SPARSE + OPTIMIZED)
# ---------------------------------------------------------

def compute_idf(corpus, vocab):
    log("Computing IDF values")

    N = len(corpus)
    df = defaultdict(int)

    for doc in corpus:
        seen = set(doc)
        for word in seen:
            if word in vocab:
                df[word] += 1

    idf = {}

    for word in vocab:
        df_val = df[word]
        idf[word] = math.log((N + 1) / (df_val + 1)) + 1

    log("IDF computation complete")
    return idf


def tfidf_vector_sparse(doc, vocab, idf):
    counts = Counter(doc)
    vec = {}
    total = len(doc)

    for word, c in counts.items():
        if word in vocab:
            tf = c / total
            vec[vocab[word]] = tf * idf[word]

    return vec


def build_tfidf_features(corpus, vocab):
    log("Building TF-IDF sparse vectors")

    idf = compute_idf(corpus, vocab)

    vectors = [tfidf_vector_sparse(doc, vocab, idf) for doc in corpus]

    log("TF-IDF feature construction complete")
    return vectors
