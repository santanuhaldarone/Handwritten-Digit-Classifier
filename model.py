import joblib
import numpy as np

class KNNModelHandler:
    def __init__(self, model_path='knn_model.pkl'):
        self.model = joblib.load(model_path)

    def predict(self, image_array):
        prediction = self.model.predict([image_array])[0]
        confidence = np.max(self.model.predict_proba([image_array]))
        return prediction, confidence
