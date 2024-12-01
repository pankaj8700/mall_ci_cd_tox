import pytest
from ml_model.train import train_model
from ml_model.predict import predict
import matplotlib.pyplot as plt
import numpy as np

def test_train_model():
    # Train the model
    model = train_model()
    assert model is not None  # Ensure the model is trained

def test_prediction():
    # Test if prediction works with the model
    sample_input = [8.3252, 41.0]
    prediction = predict(sample_input)
    assert isinstance(prediction, np.ndarray)  # Ensure prediction is a numpy array
    assert prediction.shape == (1,)  # Ensure prediction has the correct shape