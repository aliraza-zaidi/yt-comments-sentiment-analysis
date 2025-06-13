import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

        base_models = [('lr', LogisticRegression(max_iter=1000)), ('svc', SVC(probability=True))]
        
        meta_model = MultinomialNB()

        stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, n_jobs=-1)   

        pipeline = Pipeline([('tfidf', tfidf), ('stack', stack)])

        self.pipeline = pipeline
        
        return
    
    def fit_model (self):
        return self.pipeline.fit(self.X_train, self.y_train)
    
    def predict (self):
        y_pred = self.pipeline.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

        return