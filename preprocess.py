import numpy as np
from PIL import Image, ImageOps

def preprocess(image):
    image = image.convert('L')                   # Grayscale
    image = ImageOps.invert(image)               # Invert black background → white strokes
    image = image.resize((8, 8))                 # Resize
    image = np.array(image)
    image = image / 255.0 * 16                   # Match sklearn digits scale (0–16)
    return image.flatten()

