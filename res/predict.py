import numpy as np
import joblib
from res.fetch_comments import fetch_comments
from collections import Counter
from res.text_processing import TextProcessor

def predict_comments(comments, model):
    predictions = []    
    
    for comment in comments:
        prediction = model.predict([comment])[0]
        predictions.append("Positive" if prediction == 1 else "Negative")

    return predictions
