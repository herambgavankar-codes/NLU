import math
import random
from utils.logger import log

def cosine_sparse(a, b):
    dot = 0
    for i in a:
        if i in b:
            dot += a[i] * b[i]

    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))

    if na == 0 or nb == 0:
        return 0

    return dot / (na * nb)


class KNN:
    def __init__(self, k=3, sample_size=300):
        self.k = k
        self.sample_size = sample_size

    def fit(self, X, y):
        log("KNN: starting training (storing vectors)")

        if len(X) > self.sample_size:
            idx = random.sample(range(len(X)), self.sample_size)
            self.X = [X[i] for i in idx]
            self.y = [y[i] for i in idx]
            log(f"KNN: using sampled training set of size {len(self.X)}")
        else:
            self.X = X
            self.y = y
            log(f"KNN: using full training set of size {len(self.X)}")

    def predict_one(self, x):
        sims = [(cosine_sparse(x, xi), yi) for xi, yi in zip(self.X, self.y)]
        sims.sort(reverse=True)

        top = sims[:self.k]
        votes = sum(label for _, label in top)

        return 1 if votes >= self.k/2 else 0

    def predict(self, X):
        log("KNN: predicting samples")

        preds = []
        for idx, x in enumerate(X):
            if idx % 50 == 0:
                log(f"KNN: processed {idx} samples")
            preds.append(self.predict_one(x))

        log("KNN: prediction complete")
        return preds
