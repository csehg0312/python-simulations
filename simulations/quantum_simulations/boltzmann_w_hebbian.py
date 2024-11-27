import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class BoltzmannMachine:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible):
        hidden_activations = np.dot(visible, self.weights) + self.hidden_bias
        hidden_probs = self.sigmoid(hidden_activations)
        return (np.random.random(self.num_hidden) < hidden_probs).astype(int)

    def sample_visible(self, hidden):
        visible_activations = np.dot(hidden, self.weights.T) + self.visible_bias
        visible_probs = self.sigmoid(visible_activations)
        return (np.random.random(self.num_visible) < visible_probs).astype(int)

    def contrastive_hebbian_learning(self, visible_data, learning_rate, k=1):
        hidden_pos = self.sample_hidden(visible_data)
        positive_associations = np.outer(visible_data, hidden_pos)

        visible_neg = visible_data.copy()
        for _ in range(k):
            hidden_neg = self.sample_hidden(visible_neg)
            visible_neg = self.sample_visible(hidden_neg)
        negative_associations = np.outer(visible_neg, hidden_neg)

        self.weights += learning_rate * (positive_associations - negative_associations)
        self.visible_bias += learning_rate * (visible_data - visible_neg)
        self.hidden_bias += learning_rate * (hidden_pos - hidden_neg)

    def train(self, data, epochs, learning_rate, batch_size=10):
        for epoch in range(epochs):
            np.random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                for sample in batch:
                    self.contrastive_hebbian_learning(sample, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} completed")

    def reconstruct(self, visible_data):
        hidden = self.sample_hidden(visible_data)
        return self.sigmoid(np.dot(hidden, self.weights.T) + self.visible_bias)

    def get_hidden_representation(self, visible_data):
        return self.sigmoid(np.dot(visible_data, self.weights) + self.hidden_bias)

def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0  # Normalize to [0, 1]
    y = mnist.target.astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_bm(X_train, X_test, y_test):
    num_visible = X_train.shape[1]
    num_hidden = 100
    bm = BoltzmannMachine(num_visible, num_hidden)
    
    print("Training Boltzmann Machine...")
    bm.train(X_train, epochs=50, learning_rate=0.01, batch_size=10)

    print("Generating hidden representations for test data...")
    hidden_representations = np.array([bm.get_hidden_representation(x) for x in X_test])

    print("Training a simple classifier on hidden representations...")
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(hidden_representations, y_test)

    predictions = classifier.predict(hidden_representations)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on test set: {accuracy:.4f}")

    return bm, classifier

def visualize_reconstructions(bm, X_test, n_samples=5):
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples*2.5))
    for i in range(n_samples):
        original = X_test[i].reshape(28, 28)
        reconstructed = bm.reconstruct(X_test[i]).reshape(28, 28)
        
        axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructed, cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    bm, classifier = train_and_evaluate_bm(X_train, X_test, y_test)
    visualize_reconstructions(bm, X_test)

    print("\nTesting digit detection...")
    test_sample = X_test[0]
    hidden_rep = bm.get_hidden_representation(test_sample)
    predicted_digit = classifier.predict([hidden_rep])[0]
    print(f"Predicted digit: {predicted_digit}")

    plt.imshow(test_sample.reshape(28, 28), cmap='gray')
    plt.title(f"Detected Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()