from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def load_data():
    data = load_digits()
    return data.data, data.target

def plot_sample_images(X, y, n=10):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(8, 8), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
