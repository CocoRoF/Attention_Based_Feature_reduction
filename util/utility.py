import pandas as pd
import numpy as np
import torch
from numpy.linalg import norm

# 소프트맥스 함수 지정
def soft_max(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# 행렬에서 특정 값 아래이면 출력제한
def threshold_below_array(arr: np.array, threshold: float):
    return np.where(arr <= threshold, 0, arr)
  
def threshold_below_tensor(tensor: torch.Tensor, threshold: float):
    # PyTorch에서는 조건을 만족하지 않으면 0으로 설정하는 방법을 사용합니다.
    return torch.where(tensor <= threshold, torch.tensor(0.0), tensor)

def cos_sim_array(A: np.array, B: np.array):
       return np.dot(A, B)/(norm(A)*norm(B))

def cos_sim_tensor(A: torch.Tensor, B: torch.Tensor):
    return torch.dot(A, B) / (torch.norm(A) * torch.norm(B))

def array_cos_sim(array: np.array):
  num_item = len(array)
  total_array = np.empty((0, num_item))
  for i in range(num_item):
    item_list = []
    selected_vector = array[i]
    for j in range(num_item):
      temp_cos_sim = cos_sim_array(selected_vector, array[j])
      item_list.append(temp_cos_sim)
    total_array = np.vstack((total_array, np.array(item_list)))

  return total_array

def tensor_cos_sim(tensor: torch.Tensor):
    num_rows, num_cols = tensor.size()

    # 출력 텐서를 초기화합니다.
    cosine_similarity_matrix = torch.zeros(num_rows, num_cols)

    # 각 행 벡터 간의 Cosine Similarity를 계산합니다.
    for i in range(num_rows):
        for j in range(num_rows):
            # i번째 행과 j번째 행의 벡터를 가져옵니다.
            vector_i = tensor[i]
            vector_j = tensor[j]

            # Cosine Similarity 계산
            similarity = cos_sim_tensor(vector_i, vector_j)

            # 결과를 출력 텐서에 저장합니다.
            cosine_similarity_matrix[i][j] = similarity

    return cosine_similarity_matrix

def sim_result(array: np.array):
  result_lists = []
  for row in array:
      indices = np.where(row != 0)[0]  # 0이 아닌 값의 인덱스 추출
      result_lists.append(indices.tolist())

  return result_lists

def sim_result_tensor(tensor: np.array):
    result_lists = []
    for row in tensor:
        indices = torch.nonzero(row)
        indices_list = indices.squeeze().tolist()
        result_lists.append(indices_list)

    return result_lists

def filter_sublists(input_list):
    # 결과 리스트 초기화
    result = []
    result_idx = []
    for idx, sublist in enumerate(input_list):
        # 현재 서브리스트가 다른 서브리스트에 완전히 포함되는지 확인
        is_subset = any(all(item in other for item in sublist) for other in input_list if other != sublist)
        
        # 현재 서브리스트가 다른 서브리스트에 포함되지 않는 경우 결과 리스트에 추가
        if not is_subset:
            result.append(sublist)
            result_idx.append(idx)
    return result, np.array(result_idx)