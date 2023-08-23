import numpy as np
from sklearn.decomposition import PCA

#pca로구한 첫번쨰 주성분과 두번쨰 주성분을 각각 
#attention query와 attention key로 초기화 
# 랜덤하게 초기화 보다 더 좋은 어텐션 스코어를 기록할수 있을수 있다
#주성분 여러개나 다른방법 초기화도 고려가능
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def threshold_below(arr, threshold):
    return np.where(arr <= threshold, 0, arr)

class PCA_attention():
    def __init__(self, array: np.array, n_components: int, dim_info: int):
        self.array = array
        self.n_components = n_components
        self.dim_info = dim_info
        self.n_col = len(array[0])
        
        # PCA를 사용해서 데이터를 저차원으로 변환
        self.pca_data = self.apply_pca()
        
        # PCA로 변환된 데이터를 Query와 Key로 사용
        self.attention_query = self.pca_data[:dim_info]
        self.attention_key = self.pca_data[dim_info:2*dim_info]
        
    def handle_missing_values(self, array):
        return np.nan_to_num(array)  # NaN 값을 0으로 대체   
    
    def apply_pca(self):
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(self.array)
    
    def attention(self, threshold: bool = False, threshold_value: float = 0.1):
        query_matrix = self.attention_query
        key_matrix = self.attention_key
        attention_score = np.apply_along_axis(soft_max, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
        if threshold:
            return threshold_below(attention_score, threshold_value)
        else:
            return attention_score