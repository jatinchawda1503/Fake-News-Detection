import pandas as pd 
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import LancasterStemmer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


def read_data(file_path):
    return pd.read_csv(file_path)

def clean_data(data):
    data = data.fillna('')
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    return data

def preprocess(data,cols):
    data['article'] = data[cols].apply(lambda x: ' '.join(x), axis=1)
    return data 


def perform_stemming(data,st=None):
    data = re.sub('[^a-zA-Z]',' ', data)
    if not stopwords.words('english'):
        nltk.download('stopwords')
    else:
        st_eng = set(stopwords.words('english'))
    if st == 'ps':
        if not os.path.exists('./models/porterstemmer.joblib'):
            porterstemmer = PorterStemmer()
            joblib.dump(porterstemmer, './models/porterstemmer.joblib', compress=0, protocol=None, cache_size=None)
        else:
            porterstemmer = joblib.load('./models/porterstemmer.joblib')
        data = [porterstemmer.stem(word) for word in data.lower().split() if not word in st_eng]
    else:
        if not os.path.exists('./models/lancasterstemmer.joblib'):
            lancasterstemmer = LancasterStemmer()
            joblib.dump(lancasterstemmer, './models/lancasterstemmer.joblib', compress=0, protocol=None, cache_size=None)
        else:
            lancasterstemmer = joblib.load('./models/lancasterstemmer.joblib')
        data = [lancasterstemmer.stem(word) for word in data.lower().split() if not word in st_eng]
    return data



def train_split(x_data,y_data,test_size,rand):
    return train_test_split(x_data,y_data,test_size=test_size,random_state= rand)


# def vectorize_data(data,vectorizer,tot):
#     if not os.path.exists('./models/countvectorizer.joblib'):
#         countvectorizer = CountVectorizer()
#         joblib.dump(countvectorizer, './models/countvectorizer.joblib', compress=0, protocol=None, cache_size=None)
#     elif not os.path.exists('./models/tfidfvectorizer.joblib'):
#         tfidfvectorizer = TfidfVectorizer()
#         joblib.dump(tfidfvectorizer, './models/tfidfvectorizer.joblib', compress=0, protocol=None, cache_size=None)
#     else:
#         cv = joblib.load('./models/countvectorizer.joblib')
#         tfid = joblib.load('./models/tfidfvectorizer.joblib')
#         if vectorizer == 'cv' and tot == 'train':

#             data = cv.fit_transform(data)
#         elif vectorizer == 'cv' and tot == 'test':
#             data = cv.transform(data)
#         elif vectorizer == 'tfid' and tot == 'train':
#             data = tfid.fit_transform(data)
#         else:
#             data = tfid.transform(data)
#     return data


def vectorize_data(data,stage):
    if stage == 'train' and not os.path.exists('./models/countvectorizer.joblib'):  
        countvectorizer = CountVectorizer()
        countvectorizer.fit(data)
        joblib.dump(countvectorizer, './models/countvectorizer.joblib', compress=0, protocol=None, cache_size=None)
        data = countvectorizer.transform(data)
    else:
        countvectorizer = joblib.load('./models/countvectorizer.joblib')
        data = countvectorizer.transform(data)
    return data


def predict_model(xtrain,ytrain,xtest):
    if not os.path.exists('./models/logisticregression.joblib'):
        logisticregression = LogisticRegression()
        logisticregression.fit(xtrain, ytrain)
        joblib.dump(logisticregression, './models/logisticregression.joblib', compress=0, protocol=None, cache_size=None)
    else:
        logisticregression = joblib.load('./models/logisticregression.joblib')
    
    y_pred = logisticregression.predict(xtest)
    return y_pred

def get_report(actual,predicted):
    matrix = confusion_matrix(actual, predicted)
    report = classification_report(actual, predicted)
    return matrix,report


path = 'data/train.csv'
read = read_data(path)
clean = clean_data(read)
pro  = preprocess(clean,['title','author','text'])
pro['article'] = pro['article'].apply(perform_stemming,args=(''))

X = pro['article'].map(' '.join)
y = pro['label']

X_train,X_test,y_train,y_test = train_split(X,y,0.25,0)

x_train = vectorize_data(X_train,'train')

print("==values of train")

print(x_train)

x_test = vectorize_data(X_test,'test')


print("==values of test")

print(x_test)


values = predict_model(x_train,y_train,x_test)

matrix, report = get_report(y_test,values)

print(values)

print(matrix)

print(report)

