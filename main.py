import random
from dataset import load_dataset
from preprocess import preprocess_corpus
from features import (
    build_vocab,
    build_bow_features,
    build_tfidf_features
)
from experiment import run_all_models
from utils.logger import log


def split_data(X, y, test_ratio=0.2):
    log("Shuffling and splitting dataset")
    data = list(zip(X, y))
    random.shuffle(data)

    split = int(len(data) * (1 - test_ratio))
    train = data[:split]
    test = data[split:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return list(X_train), list(X_test), list(y_train), list(y_test)


def run_pipeline(feature_name, corpus, labels):
    log(f"\n==============================")
    log(f"Running pipeline using {feature_name}")
    log(f"==============================")

    vocab = build_vocab(corpus, max_features=3000)

    if feature_name == "BoW":
        vectors = build_bow_features(corpus, vocab)

    elif feature_name == "TF-IDF":
        vectors = build_tfidf_features(corpus, vocab)

    else:
        raise ValueError("Unknown feature type")

    X_train, X_test, y_train, y_test = split_data(vectors, labels)

    log(f"Training models using {feature_name}")
    results = run_all_models(X_train, X_test, y_train, y_test)

    log(f"Results for {feature_name}")
    for model_name, metrics in results.items():
        print(f"\n[{feature_name}] {model_name}")
        for k, v in metrics.items():
            print(k, ":", round(v, 4))


def main():
    log("Loading dataset")
    texts, labels = load_dataset("data/bbc-text.csv")

    log("Preprocessing text")
    corpus = preprocess_corpus(texts)

    # Run both feature extraction techniques automatically
    run_pipeline("BoW", corpus, labels)
    run_pipeline("TF-IDF", corpus, labels)

    log("All pipelines completed successfully")


if __name__ == "__main__":
    main()
