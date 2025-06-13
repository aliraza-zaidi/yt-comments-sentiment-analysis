import numpy as np
import joblib
from fetch_comments import clean_comment, is_english, fetch_comments
from collections import Counter
from text_processing import TextProcessor

def predict_comments(comments, model):
    predictions = []    
    
    for comment in comments:
        prediction = model.predict([comment])[0]
        predictions.append("Positive" if prediction == 1 else "Negative")

    return predictions

def analyze_sentiment_ratio(comments, model, preprocessor):    
    processed_comments = [preprocessor.process_comment(comment) for comment in comments]
    predictions = predict_comments(processed_comments, model)
    
    counts = Counter(predictions)
    positive = counts["Positive"]
    negative = counts["Negative"]
    total = positive + negative
    
    ratio = positive / total

    print(f"\n📊 Sentiment Analysis Summary:")
    print(f"→ Positive: {positive}")
    print(f"→ Negative: {negative}")
    print(f"→ Positive Ratio: {ratio:.2f} ({ratio*100:.1f}%)")

def find_extreme_comments(comments, model, preprocessor):
    processed_comments = [preprocessor.process_comment(c) for c in comments]
        
    probs = model.predict_proba(processed_comments)
    labels = model.classes_
        
    positive_idx = np.where(labels == 1)[0][0]
    negative_idx = np.where(labels == 0)[0][0]
    
    most_positive_idx = np.argmax(probs[:, positive_idx])
    most_negative_idx = np.argmax(probs[:, negative_idx])

    print("\n🌟 Most Positive Comment:")
    print(f"→ {comments[most_positive_idx]}")
    print(f"   Confidence: {probs[most_positive_idx, positive_idx]:.4f}")

    print("\n💔 Most Negative Comment:")
    print(f"→ {comments[most_negative_idx]}")
    print(f"   Confidence: {probs[most_negative_idx, negative_idx]:.4f}")


model = joblib.load('sentiment_model.pkl')

API_KEY = 'AIzaSyBPZoe4y2HlVGvEhZNIroSLuekKIsob-S0'
VIDEO_ID = 'XHM7xOZ_JEM'

comments = fetch_comments(VIDEO_ID, API_KEY)
tp = TextProcessor()
analyze_sentiment_ratio(comments, model, tp)
find_extreme_comments(comments, model, tp)