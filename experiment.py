from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.knn import KNN
from utils.metrics import compute_metrics
from utils.logger import log

def run_all_models(X_train, X_test, y_train, y_test):
    results = {}

    log("Training Naive Bayes")
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    log("Predicting Naive Bayes")
    results["NaiveBayes"] = compute_metrics(y_test, nb.predict(X_test))

    log("Training Logistic Regression")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    log("Predicting Logistic Regression")
    results["Logistic"] = compute_metrics(y_test, lr.predict(X_test))

    log("Training KNN")
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    log("Predicting KNN")
    results["KNN"] = compute_metrics(y_test, knn.predict(X_test))

    log("All models finished")

    return results
