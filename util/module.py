import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer

def soft_max(x):
    exp_x = np.exp(x - np.max(x))  # 지수 함수 계산 및 최대값 보정
    return exp_x / exp_x.sum(axis=0)

def threshold_below(arr, threshold):
    return np.where(arr <= threshold, 0, arr)

class Factor_attention():
    def __init__(self, array:np.array, n_factors:int, dim_info:int):
        self.array = array
        self.n_factors = n_factors
        self.dim_info = dim_info
        self.n_col = len(array[0])
        self.loadings = abs(self.factor_analyzer())
        self.attention_query = np.random.rand(self.n_factors, self.dim_info)
        self.attention_key = np.random.rand(self.n_factors, self.dim_info)
        
    def factor_analyzer(self, rotation:str="varimax"):
        analyzer = FactorAnalyzer(n_factors=self.n_factors, rotation=rotation)
        analyzer.fit(self.array)
        loadings = analyzer.loadings_
        
        return loadings
    
    def attention(self, threshold:bool=False, threshold_value:float=0.1):
        query_matrix = self.loadings.dot(self.attention_query)
        key_matrix = self.loadings.dot(self.attention_key)
        attention_score = np.apply_along_axis(soft_max, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
        if threshold:
            return threshold_below(attention_score, threshold_value)
        else:
            return attention_score