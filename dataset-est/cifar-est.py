import numpy as np
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Flatten the images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Combine train and test sets
X = np.vstack((X_train, X_test))
y = np.vstack((y_train, y_test))

# Calculate probability distribution
counts = np.bincount(y.flatten())
probabilities = counts / len(y)

# Calculate entropy
entropy = -np.sum(probabilities * np.log2(probabilities))
print("Entropy: ", entropy)