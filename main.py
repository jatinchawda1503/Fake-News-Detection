import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import LancasterStemmer


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

ps = PorterStemmer()
ls = LancasterStemmer()

def perform_stemming(data):
    data = re.sub('[^a-zA-Z]',' ', data)
    data = data.lower()
    data = data.split()
    data = [ps.stem(word) for word in data if not word in stopwords.words('english')] 
    return data



path = 'data/train.csv'
read = read_data(path)
clean = clean_data(read)
pro  = preprocess(clean,['title','author','text'])
#stemming = perform_stemming(pro)


pro['article'] = pro['article'].apply(perform_stemming)

print(pro['article'])



