from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np

def train_gmm(X, n_components=10):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    return gmm

def predict_labels(gmm, X, true_labels):
    cluster_labels = gmm.predict(X)

    # Aliniază cluster-urile cu etichetele reale
    aligned_labels = np.zeros_like(cluster_labels)
    for i in range(gmm.n_components):
        mask = (cluster_labels == i)
        if np.sum(mask) == 0:
            continue
        aligned_labels[mask] = np.bincount(true_labels[mask]).argmax()

    acc = accuracy_score(true_labels, aligned_labels)
    return aligned_labels, acc
