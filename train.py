import pandas as pd
import numpy as np
from preprocess import clean_data,preprocess_data,vectorize_data,select_features,train_split,vectorize_data,fit_model,get_prediction,get_report
import joblib



def build_model(data):
    data = clean_data(data)
    X,y =select_features(data,['title','author','text'],['label'])
    X = preprocess_data(X)
    
    X_train,X_test,y_train,y_test = train_split(X,y,0.2,0)
    X_train = vectorize_data(X_train,'train')
    X_test = vectorize_data(X_test,'test')
    model = fit_model(X_train,y_train)
    y_pred = get_prediction(model,X_test)
    matrix,report = get_report(y_test,y_pred)
    return y_pred,matrix,report


if __name__ == '__main__':
    data = pd.read_csv('./data/train.csv')
    y_pred, matrix,report = build_model(data)
    print(y_pred)
    print(matrix)
    print(report)


    
    