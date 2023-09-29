import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def normalization(X, Y):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    return scaler_X.fit_transform(X), scaler_Y.fit_transform(Y)

def standardization(X, Y):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    return scaler_X.fit_transform(X), scaler_Y.fit_transform(Y)

def data_load(data:str, preprocess_func=None, test_ratio:float=0.2, random_state:int=None):
    """
    data type = 'boston', 'house', 'home', 'ctg'
    
    'boston' is 'Boston housing dataset'. It has 13 Features(11 Numerical values, and 2 Categorical values), one Y_value(MEDV).
    
    'house' is 'Home Data for ML course' from Kaggle. It has 79 Features(), one Y_value(SalePrice)
    
    
    """
    if data == 'boston':
        path = './Data/Boston housing dataset/boston.csv'
        total = pd.read_csv(path)
        data_X_numerical = total.drop(['CHAS', 'RAD', 'MEDV'], axis=1).to_numpy()
        data_X_categorical = total[['CHAS', 'RAD']].to_numpy()
        data_Y = total['MEDV'].to_numpy().reshape(-1,1)
        
        if preprocess_func == 'normalization':
            data_X_numerical, data_Y = normalization(data_X_numerical, data_Y)
        if preprocess_func == 'standardization':
            data_X_numerical, data_Y = standardization(data_X_numerical, data_Y)
            
        X_train, X_test, y_train, y_test = train_test_split(data_X_numerical, data_Y, test_size=test_ratio, random_state=random_state)
        
    return X_train, X_test, y_train, y_test
    
    if data == 'house':
        path = './Data/home-data-for-ml-course/'
        train = pd.read_csv(path + "train.csv")
        test = pd.read_csv(path + 'test.csv')

    # if data == 'home':
    #     path = './Data/house-prices-advanced-regression-techniques/'
    #     train = pd.read_csv(path + "train.csv")
    #     test = pd.read_csv(path + 'test.csv')
        
    # if data == 'ctg':
    #     path = './Data/Cardiotocography/CTG.xls'       
        
