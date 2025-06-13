import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class SentimentModel:
    def __init__ (self, file_name):
        self.file_name = file_name
        self.data = self.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.pipeline = None 

    def load_data (self):
        return pd.read_pickle(self.file_name)
    
    def split_data (self):
        X = self.data['cleaned_comment']
        y = self.data['label']
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    def build_pipeline (self):
        tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))        
        #lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='liblinear')
        lr = MultinomialNB()
        pipeline = Pipeline([('tfidf', tfidf), ('lr', lr)])
        self.pipeline = pipeline        
        return
    
    def fit_model (self):
        return self.pipeline.fit(self.X_train, self.y_train)
    
    def predict (self):
        y_pred = self.pipeline.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        return
       
    
    def save_model (self):
        joblib.dump(self.pipeline, 'sentiment_model.pkl')
        return
    

model = SentimentModel('processed.pkl')
print("Data Loaded.....")
model.build_pipeline()
print("Pipeline Built.....")
model.fit_model()
print("Model Built....")
model.predict()
model.save_model()