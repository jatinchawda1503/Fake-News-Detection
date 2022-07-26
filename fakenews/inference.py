import pandas as pd
#import sys
#path = sys.path[-1]
import site
path = site.getsitepackages()[-1]
from fakenews.preprocess import clean_data,preprocess_data,vectorize_data,select_features,vectorize_data,get_prediction
import joblib



def make_predictions(data):
    data = clean_data(data)
    X =select_features(data,['title','author','text'])
    X = preprocess_data(X)
    X = vectorize_data(X,'test')
    model = joblib.load(path +'/fakenews/models/XGBClassifier.joblib')
    y_pred = get_prediction(model,X)
    return y_pred

if __name__ == '__main__':
    data = pd.read_csv('../data/test.csv')
    y_pred = make_predictions(data)
    print(y_pred)
