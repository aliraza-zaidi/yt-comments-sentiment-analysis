from flask import Flask, request, jsonify, render_template
from fetch_comments import fetch_comments
from text_processing import TextProcessor
from analyze import predict_comments
import joblib
import numpy as np
from collections import Counter
import re

app = Flask(__name__)
model = joblib.load("model/sentiment_model.pkl")
tp = TextProcessor()

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([\w\-]+)", url)
    return match.group(1) if match else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    video_url = data['video_url']
    video_id = extract_video_id(video_url)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    comments = fetch_comments(video_id, "AIzaSyCQbfoSsELr3qlu5XvNv97z6GERMQjfF5o")
    processed = [tp.process_comment(c) for c in comments]
    predictions = predict_comments(processed, model)

    probs = model.predict_proba(processed)
    labels = model.classes_

    positive_idx = np.where(labels == 1)[0][0]
    negative_idx = np.where(labels == 0)[0][0]

    most_positive_idx = np.argmax(probs[:, positive_idx])
    most_negative_idx = np.argmax(probs[:, negative_idx])

    counts = Counter(predictions)
    pos = counts["Positive"]
    neg = counts["Negative"]
    total = pos + neg
    ratio = pos / total if total > 0 else 0

    return jsonify({
        "count": total,
        "positive": pos,
        "negative": neg,
        "ratio": ratio,
        "most_positive": {
            "comment": comments[most_positive_idx],
            "confidence": f"{probs[most_positive_idx, positive_idx]:.4f}"
        },
        "most_negative": {
            "comment": comments[most_negative_idx],
            "confidence": f"{probs[most_negative_idx, negative_idx]:.4f}"
        }
    })

if __name__ == '__main__':
    app.run(debug=True)