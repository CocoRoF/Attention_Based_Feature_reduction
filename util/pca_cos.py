import numpy as np
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA

#pca로구한 첫번쨰 주성분과 두번쨰 주성분을 각각 
#attention query와 attention key로 초기화 

#주성분 여러개나 다른방법 초기화도 고려가능
def softmax(x):
    exp_x = np.exp(x - np.max(x))
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
        self.attention_query, self.attention_key = self.initialize_query_key_with_pca()
        
    def factor_analyzer(self, rotation:str="varimax"):
        analyzer = FactorAnalyzer(n_factors=self.n_factors, rotation=rotation)
        analyzer.fit(self.array)
        loadings = analyzer.loadings_
        return loadings
    
    def initialize_query_key_with_pca(self):
        pca = PCA(n_components=self.dim_info)
        pca_result = pca.fit_transform(self.array)
        
        # PCA 결과의 첫 번째 주성분과 두 번째 주성분을 query와 key로 사용
        query = pca_result[:, 0].reshape(self.n_factors, 1)
        key = pca_result[:, 1].reshape(self.n_factors, 1)
        return query, key
    
    def attention(self, threshold:bool=False, threshold_value:float=0.1):
        query_matrix = self.loadings.dot(self.attention_query)
        key_matrix = self.loadings.dot(self.attention_key)
        attention_score = np.apply_along_axis(softmax, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
        if threshold:
            return threshold_below(attention_score, threshold_value)
        else:
            return attention_score


