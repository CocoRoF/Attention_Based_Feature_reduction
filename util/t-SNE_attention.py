import numpy as np
from sklearn.manifold import TSNE

# 비선형 차원축소방법 t-sne 로 만들어
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def threshold_below(arr, threshold):
    return np.where(arr <= threshold, 0, arr)

class TSNE_attention():
    def __init__(self, array: np.array, n_components: int, dim_info: int):
        self.array = array
        self.n_components = n_components
        self.dim_info = dim_info
        self.n_col = len(array[0])
        
        # t-SNE를 사용해서 데이터를 저차원으로 변환
        self.tsne_data = self.apply_tsne()
        
        # t-SNE로 변환된 데이터를 Query와 Key로 사용
        self.attention_query = self.tsne_data[:dim_info]
        self.attention_key = self.tsne_data[dim_info:2*dim_info]
        
    def handle_missing_values(self, array):
        return np.nan_to_num(array)  # NaN 값을 0으로 대체   
    
    def apply_tsne(self):
        tsne = TSNE(n_components=self.n_components)
        return tsne.fit_transform(self.array)
    
    def attention(self, threshold: bool = False, threshold_value: float = 0.1):
        query_matrix = self.attention_query
        key_matrix = self.attention_key
        attention_score = np.apply_along_axis(softmax, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
        if threshold:
            return threshold_below(attention_score, threshold_value)
        else:
            return attention_score

