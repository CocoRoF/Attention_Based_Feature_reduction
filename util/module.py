import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from factor_analyzer import FactorAnalyzer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from util.utility import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# array를 받아서 Columns들을 Vector로 변경.
class columns_to_vector():
    def __init__(self, array:np.array, n_factors:int=None):
        self.array = array
        if n_factors:
          self.n_factors = n_factors
        else:
          self.n_factors = len(array[0])
        self.n_col = len(array[0])
        
    def factor(self, rotation:str='varimax'): 
        analyzer = FactorAnalyzer(n_factors=self.n_factors, rotation=rotation)
        analyzer.fit(self.array)
        loadings = analyzer.loadings_
        return loadings

class Factor_attention():
  def __init__(self, array:np.array, n_factors:int=None, dim_information:int=16):
    self.array = array
    if n_factors:
          self.n_factors = n_factors
    else:
      self.n_factors = len(array[0])
    self.dim_info = dim_information
    self.n_col = len(array[0])

  def col_to_vec(self, method:str = 'factor', rotation:str = 'varimax', threshold:float = 0.5):
    if method == 'factor':
      self.column_vector = columns_to_vector(self.array, self.n_factors).factor(rotation = rotation)
      self.cos_sim = array_cos_sim(self.column_vector)
      self.under_cos_sim = threshold_below_array(self.cos_sim, threshold)
      self.total_result = sim_result(self.under_cos_sim)
      self.result, self.result_idx = filter_sublists(self.total_result)

    unique_list = []
    for sublist in self.result:
        if sublist not in unique_list:
            unique_list.append(sublist)
    self.result = unique_list
    
    self.selected_array_list = []
    for i in self.result:
      selected_array = self.array[:, i]
      self.selected_array_list.append(selected_array)
    self.selected_raw_tensor_list = [torch.Tensor(arr).double().to(device) for arr in self.selected_array_list]
    
    self.selected_value_list = []
    for i in self.result:
      selected_value = self.column_vector[i, :]
      self.selected_value_list.append(selected_value)
    self.selected_factor_tensor_list = [torch.Tensor(arr).double().to(device) for arr in self.selected_value_list]

      
def min_relation(tensor: torch.Tensor, n_factors:int=None, method:str ='factor', rotation:str = 'varimax'):
  temp_array = tensor.detach().numpy()
  if n_factors:
    n_factor = n_factors
  else:
    n_factor = temp_array.shape[-1]
    
  column_vector = columns_to_vector(temp_array, n_factor).factor(rotation = rotation)
  cos_sim = array_cos_sim(column_vector)
  abs_cos_sim = abs(cos_sim)
  sum_cos_sim = abs_cos_sim.sum() - n_factor
  return torch.tensor(sum_cos_sim)
  
class Attention(nn.Module):
  def __init__(self, n_factor, info_dim:int = 512):
    super(Attention, self).__init__()
    self.attention_query = nn.Parameter(torch.randn(n_factor, info_dim).double().to(device), requires_grad=True)
    self.attention_key = nn.Parameter(torch.randn(n_factor, info_dim).double().to(device), requires_grad=True)
    self.attention_value = nn.Parameter(torch.randn(n_factor, info_dim).double().to(device), requires_grad=True)
    self.output_linear_1 = nn.Linear(info_dim, 256).to(device)
    self.output_linear_2 = nn.Linear(256, 128).to(device)
    self.output_linear_3 = nn.Linear(128, 64).to(device)
    self.output_linear_4 = nn.Linear(64, 1).to(device)
        
  def forward(self, Factor_Attention: Factor_attention):
    input_1 = Factor_Attention.selected_raw_tensor_list
    input_2 = Factor_Attention.selected_factor_tensor_list
    self.score_weight_list = []
    for variable in input_2:      
      if variable.shape[0] == 1:
        score_weight = torch.ones(1, 1).double().to(device)
        self.score_weight_list.append(score_weight)
        
      else:
        query_result = torch.matmul(variable, self.attention_query).to(device)
        key_result = torch.matmul(variable, self.attention_key).to(device)
        value_result = torch.matmul(variable, self.attention_value).to(device)
        attention_filter = torch.matmul(query_result, key_result.T).to(device)
        attention_score = nn.Softmax(dim=-1)(attention_filter).to(device)
        attention_context = torch.matmul(attention_score, value_result).to(device)
        hidden_1 = self.output_linear_1(attention_context.float()).to(device)
        hidden_2 = self.output_linear_2(hidden_1.float()).to(device)
        hidden_3 = self.output_linear_3(hidden_2.float()).to(device)
        score_weight = self.output_linear_4(hidden_3.float()).to(device)
        self.score_weight_list.append(score_weight.double().to(device))
      
    if len(self.score_weight_list) != len(input_1):
      print('Invalid Length Error.')
    
    self.total_result = torch.empty(Factor_Attention.array.shape[0], 0).to(device)
    for num in range(len(self.score_weight_list)):
      result = torch.matmul((input_1[num]).to(device), self.score_weight_list[num].to(device)).to(device)
      self.total_result = torch.cat((self.total_result, result), dim=1).to(device)
    
    return self.total_result
        


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
