import numpy as np
from sklearn.manifold import TSNE

def soft_max(x):
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
        
        # t-SNE를 사용해서 데이터의 임베딩을 얻음
        self.embeddings = self.apply_tsne()
        
        # 얻어진 임베딩을 Query와 Key로 사용
        self.attention_query = self.embeddings[:dim_info]
        self.attention_key = self.embeddings[dim_info:2*dim_info]
        
    def apply_tsne(self):
        if self.dim_info <= 4:
            tsne = TSNE(n_components=self.dim_info)
        else:
            tsne = TSNE(n_components=self.dim_info, method='exact')
        return tsne.fit_transform(self.array)


    def attention(self, threshold: bool = False, threshold_value: float = 0.1):
        query_matrix = self.embeddings[:self.dim_info]
        key_matrix = self.embeddings[self.dim_info:2*self.dim_info]
        
        attention_score = query_matrix.dot(key_matrix.T)
        attention_score = np.apply_along_axis(soft_max, axis=1, arr=attention_score)
        
        if threshold:
            return threshold_below(attention_score, threshold_value)
        else:
            return attention_score

#Factor_attention에서는 loadings 값을 통해 attention을 계산하는 반면,
#TSNE_attention에서는 t-SNE의 결과인 임베딩 값을 사용하여 attention을 계산하시면 됩니다.

