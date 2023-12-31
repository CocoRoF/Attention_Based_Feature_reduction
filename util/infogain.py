import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from util.utility import *

# 전체 데이터의 Array로부터 Information을 추출하는 방법들에 대해서만 기술합니다.
# 최종 결과는 전체 데이터의 Feature를 n차원의 Vector로 표현한 Matrix가 됩니다.

# Input = Data_X_Array, output = Information_matrix of X_Data's Features(array)

class Feature_infogain():
    def __init__(self, X_Data:np.array):
        self.X_Data = X_Data
        self.num_feature = X_Data.shape[-1]
        self.num_sample = X_Data.shape[0]
    
    # 특정 Feature의 정보를, 다른 Feature와의 관계성으로 표현한다고 생각
    def Corr(self):
        return np.corrcoef(self.X_Data.T)
    
    # 특정 Feature의 정보를, 잠재된 Factor와의 관계성으로 표현한다고 생각
    def FA(self, n_factor:int=None, rotation:str='varimax'):
        if n_factor == None:
            self.n_factor = self.num_feature
        else:
            self.n_factor = n_factor
        analyzer = FactorAnalyzer(n_factors=self.n_factor, rotation=rotation)
        analyzer.fit(self.X_Data)
        return analyzer.loadings_
    
            
        

    
    
