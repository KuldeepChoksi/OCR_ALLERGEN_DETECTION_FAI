import numpy as np
from PIL import Image

def load_image(image_path):
    # load
    return Image.open(image_path).convert('L')

def binarize(image, threshold=128):
    # convert to grey
    return image.point(lambda p: 255 if p > threshold else 0)

def segment_characters(image):
    # basic segmentation
    arr = np.array(image)
    char_width = 20
    return [arr[:, i:i+char_width] for i in range(0, arr.shape[1], char_width)]