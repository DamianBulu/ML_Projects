from data_loader import load_data, plot_sample_images
from model import train_gmm, predict_labels

def main():
    X, y = load_data()
    plot_sample_images(X, y)

    gmm = train_gmm(X, n_components=10)
    predicted, accuracy = predict_labels(gmm, X, y)

    print(f"Accuracy (after label alignment): {accuracy:.2f}")
#Fisier modificat
if __name__ == "__main__":
    main()
