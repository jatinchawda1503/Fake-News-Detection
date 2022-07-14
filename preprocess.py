import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import LancasterStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def clean_data(data):
    data = data.fillna('')
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    return data

def select_features(data,colx,coly=None):
    X = data[colx].apply(lambda x: ' '.join(x), axis=1)
    if coly is None:
        return X
    else:
        y = data[coly]
    return X,y

def preprocess_data(data):
    data = data.str.replace('[^a-zA-Z0-9]', ' ',regex=True)
    if not stopwords.words('english'):
        nltk.download('stopwords')
    else:
        st_eng = set(stopwords.words('english'))   
    if not os.path.exists('./models/lancasterstemmer.joblib'):
            lancasterstemmer = LancasterStemmer()
            joblib.dump(lancasterstemmer, './models/lancasterstemmer.joblib', compress=0, protocol=None, cache_size=None)
            data = data.apply(lambda words: ' '.join(lancasterstemmer.stem(word.lower()) for word in words.split() if word not in st_eng))
    else:
        lancasterstemmer = joblib.load('./models/lancasterstemmer.joblib')
        data = data.apply(lambda words: ' '.join(lancasterstemmer.stem(word.lower()) for word in words.split() if word not in st_eng))
    return data



def train_split(x_data,y_data,test_size,rand):
    return train_test_split(x_data,y_data,test_size=test_size,random_state= rand)


def vectorize_data(data,stage):
    if stage == 'train' and not os.path.exists('./models/TfidfVectorizer.joblib'):  
        vectorizer = TfidfVectorizer()
        vectorizer.fit(data)
        joblib.dump(vectorizer, './models/TfidfVectorizer.joblib', compress=0, protocol=None, cache_size=None)
        data = vectorizer.transform(data)
    else:
        vectorizer = joblib.load('./models/TfidfVectorizer.joblib')
        data = vectorizer.transform(data)
    return data

def fit_model(x_data,y_data):
    if not os.path.exists('./models/logisticregression.joblib'):
        model = LogisticRegression()
        model.fit(x_data, y_data)
        joblib.dump(model, './models/logisticregression.joblib', compress=0, protocol=None, cache_size=None)
    else:
        model = joblib.load('./models/logisticregression.joblib')
    return model

def get_prediction(model,data):
    y_pred = model.predict(data)
    return y_pred

def get_report(actual,predicted):
    matrix = confusion_matrix(actual, predicted)
    report = classification_report(actual, predicted)
    return matrix,report
    

