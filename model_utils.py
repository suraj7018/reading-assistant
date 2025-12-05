import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

MODEL_PATH = "difficulty_model.pkl"

def train_initial_model():
    """Trains a dummy initial model to bootstrap the adapt agent."""
    # Features: [error_rate (0-1), wpm (0-200), focus_score (0-1)]
    # Target: difficulty_index (0-1)
    
    # Heuristic data generation for cold start
    X = np.array([
        [0.0, 150, 0.9], # Low errors, high speed, high focus -> Increase difficulty
        [0.1, 100, 0.7], # Moderate -> Moderate difficulty
        [0.3, 50,  0.4], # High errors, low speed, low focus -> Decrease difficulty
        [0.5, 30,  0.2], # Very high errors -> Very low difficulty
    ])
    y = np.array([0.9, 0.6, 0.3, 0.1])

    model = LinearRegression()
    model.fit(X, y)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Initial model trained and saved to {MODEL_PATH}")
    return model

def load_model():
    """Loads the model or trains it if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return train_initial_model()

def predict_difficulty(model, error_rate, wpm, focus_score):
    """Predicts the next difficulty index."""
    input_features = np.array([[error_rate, wpm, focus_score]])
    prediction = model.predict(input_features)[0]
    # Clip result to valid range [0.0, 1.0]
    return max(0.0, min(1.0, prediction))
