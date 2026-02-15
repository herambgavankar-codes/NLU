from collections import defaultdict
import math
from utils.logger import log

class NaiveBayes:
    def fit(self, X, y):
        log("NaiveBayes: starting training")

        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_words = defaultdict(int)

        for xi, yi in zip(X, y):
            self.class_counts[yi] += 1
            for i, v in xi.items():
                self.feature_counts[yi][i] += v
                self.total_words[yi] += v

        self.vocab_size = len({k for x in X for k in x})
        self.classes = list(self.class_counts.keys())

        log(f"NaiveBayes: training complete | classes={self.classes} vocab_size={self.vocab_size}")

    def predict_one(self, x):
        scores = {}

        for c in self.classes:
            log_prob = math.log(self.class_counts[c] + 1)

            for i, v in x.items():
                num = self.feature_counts[c][i] + 1
                den = self.total_words[c] + self.vocab_size
                prob = num / den
                log_prob += v * math.log(prob)

            scores[c] = log_prob

        return max(scores, key=scores.get)

    def predict(self, X):
        log("NaiveBayes: predicting samples")
        preds = [self.predict_one(x) for x in X]
        log("NaiveBayes: prediction complete")
        return preds
