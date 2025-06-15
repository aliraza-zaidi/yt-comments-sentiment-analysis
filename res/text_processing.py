import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    def __init__ (self):        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def load_data (self, filename):
        self.data = pd.read_pickle(filename)
        
    def clean_text (self, text):
        text = text.lower()
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)
    
    def process_data (self):
        self.data["cleaned_comment"] = self.data["comment"].apply(self.clean_text)
        
    def process_comment (self, comment):
        return self.clean_text(comment)
    
    def save_data (self):
        self.data.to_pickle('data/processed.pkl')

tp = TextProcessor()
tp.load_data("data/comments.pkl")
tp.process_data()
tp.save_data()