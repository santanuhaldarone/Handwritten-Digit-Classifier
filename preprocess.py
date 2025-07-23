import numpy as np
from PIL import Image, ImageOps

def preprocess_mnist_style(image_data):
    
    image = Image.fromarray(image_data).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = ImageOps.autocontrast(image)
    image = np.array(image) / 255.0
    return image.reshape(1, -1)
