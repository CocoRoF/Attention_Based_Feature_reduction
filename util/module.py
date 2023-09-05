import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from util.utility import *

# array를 받아서 Columns들을 Vector로 변경.
class columns_to_vector():
    def __init__(self, array:np.array, n_factors:int, dim_info:int):
        self.array = array
        self.n_factors = n_factors
        self.dim_info = dim_info
        self.n_col = len(array[0])
        
    def factor(self, rotation:str='varimax'): 
        analyzer = FactorAnalyzer(n_factors=self.n_factors, rotation=rotation)
        analyzer.fit(self.array)
        loadings = analyzer.loadings_
        return loadings

class Factor_attention():
  def __init__(self, array:np.array, n_factors:int, dim_infomation:int):
    self.array = array
    self.n_factors = n_factors
    self.dim_info = dim_infomation
    self.n_col = len(array[0])

  def col_to_vec(self, method:str = 'factor', rotation:str = 'varimax', threshold:float = 0.5):
    if method == 'factor':
      self.column_vector = columns_to_vector(self.array, self.n_factors, self.dim_info).factor(rotation = rotation)
      self.cos_sim = array_cos_sim(self.column_vector)
      self.under_cos_sim = threshold_below(self.cos_sim, threshold)
      self.total_result = sim_result(self.under_cos_sim)
      self.result, self.result_idx = filter_sublists(self.total_result)



# class columns_to_vector():
#     def __init__(self, array:np.array, n_factors:int, dim_info:int):
#         self.array = array
#         self.n_factors = n_factors
#         self.dim_info = dim_info
#         self.n_col = len(array[0])
        
#     def factor(self, rotation:str='varimax'): 
#         analyzer = FactorAnalyzer(n_factors=self.n_factors, rotation=rotation)
#         analyzer.fit(self.array)
#         loadings = analyzer.loadings_
#         return loadings

# def factor_to_result(array: np.array, threshold: int = 0.5):
#     sim_array = array_cos_sim(array)
#     temp_array = threshold_below(sim_array, threshold)
#     result = sim_result(temp_array)
#     return result

# # Factor Analysis를 통해 Column의 값을 Factor를 통한 Vector로 만들고, 이를 활용해 어텐션 진행.
# # But 현재 상태에서는 그냥 랜덤한 값을 이용해서 출력할 뿐 이것이 좋다고 말할 수는 없는 상황임
# # 이에대해서는 조금 더 고민해볼 필요가 있을 것으로 보임 ...
# # class Factor_attention():
# #     def __init__(self, array:np.array, n_factors:int, dim_info:int):
# #         self.array = array
# #         self.n_factors = n_factors
# #         self.dim_info = dim_info
# #         self.n_col = len(array[0])
# #         self.loadings = abs(self.factor_analyzer())
# #         self.attention_query = np.random.rand(self.n_factors, self.dim_info)
# #         self.attention_key = np.random.rand(self.n_factors, self.dim_info)
        
# #     def factor_analyzer(self, rotation:str="varimax"):
# #         analyzer = FactorAnalyzer(n_factors=self.n_factors, rotation=rotation)
# #         analyzer.fit(self.array)
# #         loadings = analyzer.loadings_
        
# #         return loadings

# #     def attention(self, threshold:bool=False, threshold_value:float=0.1):
# #         query_matrix = self.loadings.dot(self.attention_query)
# #         key_matrix = self.loadings.dot(self.attention_key)
# #         attention_score = np.apply_along_axis(soft_max, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
# #         if threshold:
# #             return threshold_below(attention_score, threshold_value)
# #         else:
# #             return attention_score

# # 비선형 차원축소방법 t-sne 로 만들어
# class TSNE_attention():
#     def __init__(self, array: np.array, n_components: int, dim_info: int):
#         self.array = array
#         self.n_components = n_components
#         self.dim_info = dim_info
#         self.n_col = len(array[0])
        
#         # t-SNE를 사용해서 데이터를 저차원으로 변환
#         self.tsne_data = self.apply_tsne()
        
#         # t-SNE로 변환된 데이터를 Query와 Key로 사용
#         self.attention_query = self.tsne_data[:dim_info]
#         self.attention_key = self.tsne_data[dim_info:2*dim_info]
        
#     def handle_missing_values(self, array):
#         return np.nan_to_num(array)  # NaN 값을 0으로 대체   
    
#     def apply_tsne(self):
#         tsne = TSNE(n_components=self.n_components)
#         return tsne.fit_transform(self.array)
    
#     def attention(self, threshold: bool = False, threshold_value: float = 0.1):
#         query_matrix = self.attention_query
#         key_matrix = self.attention_key
#         attention_score = np.apply_along_axis(softmax, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
#         if threshold:
#             return threshold_below(attention_score, threshold_value)
#         else:
#             return attention_score

# # pca로구한 첫번쨰 주성분과 두번쨰 주성분을 각각 
# # attention query와 attention key로 초기화 
# # 랜덤하게 초기화 보다 더 좋은 어텐션 스코어를 기록할수 있을수 있다
# # 주성분 여러개나 다른방법 초기화도 고려가능        
# class PCA_attention():
#     def __init__(self, array: np.array, n_components: int, dim_info: int):
#         self.array = array
#         self.n_components = n_components
#         self.dim_info = dim_info
#         self.n_col = len(array[0])
        
#         # PCA를 사용해서 데이터를 저차원으로 변환
#         self.pca_data = self.apply_pca()
        
#         # PCA로 변환된 데이터를 Query와 Key로 사용
#         self.attention_query = self.pca_data[:dim_info]
#         self.attention_key = self.pca_data[dim_info:2*dim_info]
        
#     def handle_missing_values(self, array):
#         return np.nan_to_num(array)  # NaN 값을 0으로 대체   
    
#     def apply_pca(self):
#         pca = PCA(n_components=self.n_components)
#         return pca.fit_transform(self.array)
    
#     def attention(self, threshold: bool = False, threshold_value: float = 0.1):
#         query_matrix = self.attention_query
#         key_matrix = self.attention_key
#         attention_score = np.apply_along_axis(soft_max, axis=1, arr=(query_matrix.dot(key_matrix.T)))
        
#         if threshold:
#             return threshold_below(attention_score, threshold_value)
#         else:
#             return attention_score
