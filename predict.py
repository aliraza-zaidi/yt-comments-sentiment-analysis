import numpy as np
import joblib

def predict_comment(comment):
    prediction = model.predict([comment])[0]
    return "Positive" if prediction == 1 else "Negative"


def top_features(pipeline, n=20):
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['lr']
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    top_pos = sorted(zip(coefs, feature_names), reverse=True)[:n]
    top_neg = sorted(zip(coefs, feature_names))[:n]
    print("Top Positive Words:")
    for coef, feat in top_pos:
        print(f"{feat}: {coef:.4f}")
    print("\nTop Negative Words:")
    for coef, feat in top_neg:
        print(f"{feat}: {coef:.4f}")


def explain_prediction(model, comment):

    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['lr']

    feature_names = vectorizer.get_feature_names_out()
    tfidf_vector = vectorizer.transform([comment])
    coef = classifier.coef_[0]

    contribution = tfidf_vector.toarray()[0] * coef
    
    top_index = np.argmax(np.abs(contribution))
    top_word = feature_names[top_index]
    word_contribution = contribution[top_index]
    
    pred = classifier.predict(tfidf_vector)[0]
    print(f"Predicted sentiment: {pred}")
    print(f"Top contributing word: '{top_word}' (score: {word_contribution:.4f})")


model = joblib.load('sentiment_model.pkl')
explain_prediction(model, "Content and analysis is very bad")