import numpy as np
import joblib

def predict_comment (comment):
    prediction = model.predict([comment])[0]
    return "Positive" if prediction == 1 else "Negative"

# def contributing_word (model, comment):

#     vectorizer = model.named_steps['tfidf']
#     classifier = model.named_steps['lr']

#     feature_names = vectorizer.get_feature_names_out()
#     tfidf_vector = vectorizer.transform([comment])
#     coef = classifier.coef_[0]

#     contribution = tfidf_vector.toarray()[0] * coef
    
#     top_index = np.argmax(np.abs(contribution))
#     top_word = feature_names[top_index]
#     word_contribution = contribution[top_index]
    
#     pred = classifier.predict(tfidf_vector)[0]
#     print(f"Predicted sentiment: {pred}")
#     print(f"Top contributing word: '{top_word}' (score: {word_contribution:.4f})")
import numpy as np

def contributing_word(model, comment):
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['lr']  # Assuming pipeline step name is 'nb'

    feature_names = vectorizer.get_feature_names_out()
    tfidf_vector = vectorizer.transform([comment])
    log_probs = classifier.feature_log_prob_

    # Compute log-prob difference between positive and negative class
    # Assumes binary classification: class 0 (negative), class 1 (positive)
    class_difference = log_probs[1] - log_probs[0]  # shape: (n_features,)
    
    tfidf_array = tfidf_vector.toarray()[0]
    contribution = tfidf_array * class_difference

    top_index = np.argmax(np.abs(contribution))
    top_word = feature_names[top_index]
    word_contribution = contribution[top_index]
    
    pred = classifier.predict(tfidf_vector)[0]
    print(f"Predicted sentiment: {pred}")
    print(f"Top contributing word: '{top_word}' (score: {word_contribution:.4f})")


model = joblib.load('sentiment_model.pkl')
comment = "Did not expect you to openly criticize so many different elites. Utterly disgusted."
contributing_word(model, comment)