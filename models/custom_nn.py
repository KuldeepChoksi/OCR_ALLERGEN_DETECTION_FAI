import numpy as np
import pickle
import os
from PIL import Image


class SimpleCNN:

    def __init__(self, input_size=784, hidden_size=64, output_size=36):
        # Initialize weights with proper dimensions
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

        # Character mapping (A-Z, 0-9)
        self.char_map = {i: c for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp = np.exp(x - np.max(x))  # Numerical stability
        return exp / exp.sum()

    def forward(self, x):
        # Flatten input image
        x = x.flatten()

        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)

    def train(self, X, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            for i in range(len(X)):
                # Forward pass
                x = X[i]
                y_true = np.zeros(36)
                y_true[y[i]] = 1  # One-hot encoding

                probs = self.forward(x)

                # Calculate loss (cross-entropy)
                loss = -np.log(probs[y[i]])
                total_loss += loss

                # Backward pass
                error = probs - y_true

                # Gradients
                dW2 = np.outer(self.a1, error)
                db2 = error.copy()

                dhidden = np.dot(self.W2, error) * (self.z1 > 0)  # ReLU derivative
                dW1 = np.outer(x, dhidden)
                db1 = dhidden.copy()

                # Update weights
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2

                # Track accuracy
                if np.argmax(probs) == y[i]:
                    correct += 1

            # Print training progress
            accuracy = correct / len(X)
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2%}")

    def predict(self, x):
        probs = self.forward(x)
        return self.char_map[np.argmax(probs)]

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'char_map': self.char_map
            }, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.W1 = data['W1']
        model.b1 = data['b1']
        model.W2 = data['W2']
        model.b2 = data['b2']
        model.char_map = data['char_map']
        return model


# Helper function to load training data
def load_training_data(data_dir):
    X, y = [], []
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    for i, char in enumerate(chars):
        img_path = os.path.join(data_dir, f"{char}.png")
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert('L')) / 255.0
            X.append(img)
            y.append(i)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    model = SimpleCNN()
    X_train, y_train = load_training_data("../data/train")
    model.train(X_train, y_train, epochs=15)
    model.save("../models/ocr_model.pkl")