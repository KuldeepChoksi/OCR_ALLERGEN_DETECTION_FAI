import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import pickle  # Needed for model serialization


class OCRModel:
    def __init__(self):
        self.model = None
        self.char_map = {}

    def train(self, data_dir):
        X, y = [], []

        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
            img_path = f"{data_dir}/{char}.png"
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')
                X.append(np.array(img).flatten())
                y.append(i)
                self.char_map[i] = char

        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(X, y)

    def predict(self, char_image):
        # Predict a single character
        if self.model is None:
            raise ValueError("Model not trained")
        pred = self.model.predict([char_image.flatten()])[0]
        return self.char_map[pred]

    def save(self, file_path):
        # Save
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'char_map': self.char_map
            }, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.model = data['model']
        model.char_map = data['char_map']
        return model