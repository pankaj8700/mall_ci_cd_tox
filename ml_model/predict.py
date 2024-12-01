import joblib
import numpy as np

def predict(input_features):
    model = joblib.load('ml_model/kmeans.pkl')
    
    # Ensure input_features is a 2D array
    input_features = np.array(input_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Return the prediction
    return prediction