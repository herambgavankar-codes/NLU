import matplotlib.pyplot as plt

def plot_model_comparison(results):
    names = list(results.keys())
    acc = [results[m]["accuracy"] for m in names]

    plt.figure()
    plt.bar(names, acc)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.show()
