import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text (text):
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

data = pd.read_pickle("comments.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

data["cleaned_comment"] = data["comment"].apply(clean_text)

