import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# dataloader는 특정 데이터를 받아 Array 형태로 반환한다.
# 이 때 X는 Numerical Values만을 반환하며, 필요에 따라 정규화나, 표준화를 실시할 수 있다.
# 아직 Categorical Values에 대한 처리는 어떻게 해야할지 결정하지 않았다.

# input = Data Name / output = Data Array(X and Y)

def normalization(X, Y):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    return scaler_X.fit_transform(X), scaler_Y.fit_transform(Y)

def standardization(X, Y):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    return scaler_X.fit_transform(X), scaler_Y.fit_transform(Y)

def data_load(data:str, preprocess_func=None, split:bool=False, test_ratio:float=0.2, random_state:int=42):
    """
    data type = 'boston', 'house', 'home', 'ctg'
    
    'boston' is 'Boston housing dataset'. It has 13 Features(11 Numerical values, and 2 Categorical values), one Y_value(MEDV).
    
    'house' is 'Home Data for ML course' from Kaggle. It has 57 Features(20 Numerical values, and 36 Categorical values), one Y_value(SalePrice)
    
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
        
        if split:
            return X_train, X_test, y_train, y_test
        else:
            return data_X_numerical, data_Y
    
    if data == 'house':
        path = './Data/home-data-for-ml-course/'
        train = pd.read_csv(path + "train_dropna.csv")
        test = pd.read_csv(path + 'test_dropna.csv')
        
        data_X_numerical = train.select_dtypes(include=['int', 'float'])
        data_X_numerical = data_X_numerical.drop(['SalePrice'], axis=1).to_numpy()
        data_X_categorical = train.select_dtypes(include=['object']).to_numpy()
        data_Y = train['SalePrice'].to_numpy().reshape(-1,1)
        
        if preprocess_func == 'normalization':
            data_X_numerical, data_Y = normalization(data_X_numerical, data_Y)
        if preprocess_func == 'standardization':
            data_X_numerical, data_Y = standardization(data_X_numerical, data_Y)
            
        X_train, X_test, y_train, y_test = train_test_split(data_X_numerical, data_Y, test_size=test_ratio, random_state=random_state)
        
        if split:
            return X_train, X_test, y_train, y_test
        else:
            return data_X_numerical, data_Y

    # if data == 'home':
    #     path = './Data/house-prices-advanced-regression-techniques/'
    #     train = pd.read_csv(path + "train.csv")
    #     test = pd.read_csv(path + 'test.csv')
        
    # if data == 'ctg':
    #     path = './Data/Cardiotocography/CTG.xls'       
        
