import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset
def entropy_mnist():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist['data']
    y = mnist['target']

    # Calculate probability distribution
    counts = np.bincount(y.astype(int))
    probabilities = counts / len(y)

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    # print("Entropy: ", entropy)
    return entropy