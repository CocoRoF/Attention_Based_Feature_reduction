import pandas as pd
import numpy as np
import torch
from numpy.linalg import norm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Helper functions for EM-based FA

def e_step(X, loadings, unique_variances):
    """
    E-step of the EM algorithm for FA.
    """
    X, loadings, unique_variances = X.double().to(device), loadings.double().to(device), unique_variances.double().to(device)

    inv_diag_unique_variances = 1.0 / unique_variances
    inv_sigma = torch.diag(inv_diag_unique_variances) - torch.mm(
        torch.mm(loadings.t(), torch.diag(inv_diag_unique_variances)),
        torch.mm(loadings, torch.inverse(torch.eye(loadings.shape[1], device=device, dtype=torch.double) +
                                         torch.mm(loadings.t(), torch.mm(torch.diag(inv_diag_unique_variances), loadings))))
    )

    latent_mean = torch.mm(torch.mm(X, torch.diag(inv_diag_unique_variances)), torch.mm(loadings, inv_sigma))

    latent_covariance = torch.eye(loadings.shape[1], device=device, dtype=torch.double) - torch.mm(
        torch.mm(loadings.t(), torch.diag(inv_diag_unique_variances)),
        torch.mm(loadings, inv_sigma)
    ) + torch.mm(inv_sigma, torch.mm(latent_mean.t(), latent_mean))

    return latent_mean, latent_covariance

def m_step(X, latent_mean, latent_covariance):
    """
    M-step of the EM algorithm for FA.
    """
    X, latent_mean, latent_covariance = X.double(), latent_mean.double(), latent_covariance.double()

    loadings = torch.mm(torch.mm(X.t(), latent_mean), torch.inverse(latent_covariance))

    diag_residual = torch.diag(torch.mm(X.t(), X)) - torch.diag(torch.mm(loadings, torch.mm(latent_mean.t(), X)))
    unique_variances = diag_residual / X.shape[0]

    return loadings, unique_variances

def em_factor_analysis(X, n_components, max_iter=100):
    """
    Factor Analysis using the EM algorithm.
    """
    n_features = X.shape[1]

    # Initialize parameters
    loadings = torch.randn(n_features, n_components, device=device, dtype=torch.double)
    unique_variances = torch.ones(n_features, device=device, dtype=torch.double)

    for i in range(max_iter):
        # E-step
        latent_mean, latent_covariance = e_step(X, loadings, unique_variances)

        # M-step
        loadings, unique_variances = m_step(X, latent_mean, latent_covariance)

    return loadings, unique_variances

def varimax_rotation(loadings, gamma=1.0, max_iter=5, tolerance=1e-6):
    """
    Apply Varimax rotation to the loadings matrix.
    """
    loadings = loadings.double()

    n, m = loadings.shape
    rotated_loadings = loadings.clone()
    rotation_matrix = torch.eye(m, device=device, dtype=torch.double)

    for _ in range(max_iter):
        d = torch.mm(rotated_loadings, torch.mm(torch.diag(torch.sum(rotated_loadings ** 2, dim=0) ** (-gamma/2)), rotated_loadings.t()))
        print(torch.isnan(rotated_loadings).any())
        print(torch.isinf(rotated_loadings).any())
        u, s, v = torch.svd(torch.mm(d, rotated_loadings - (1.0 / n) * torch.mm(rotated_loadings, torch.diag(torch.sum(rotated_loadings, dim=0)))))
        rotation_matrix_new = torch.mm(u, v)
        rotated_loadings = torch.mm(loadings, rotation_matrix_new)

        if torch.norm(rotation_matrix_new - rotation_matrix) < tolerance:
            break

        rotation_matrix = rotation_matrix_new

    return rotated_loadings

def cosine_similarity_loss(tensor: torch.Tensor):
    similarity_matrix = F.cosine_similarity(tensor.unsqueeze(0), tensor.unsqueeze(1), dim=2)
    loss = torch.mean(similarity_matrix)  # 평균 Cosine Similarity
    return similarity_matrix

def cosine_similarity_matrix(tensor: torch.Tensor):
    n, m = tensor.shape
    if n != m:
        raise ValueError("The input tensor must be square (n x n).")

    result = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            result[i][j] = F.cosine_similarity(tensor[i], tensor[j], dim=0)

    return result

def factor_analysis(tensor: torch.Tensor, num_iter:int = 100):
    loadings, unique_variances = em_factor_analysis(tensor, tensor.shape[-1], max_iter=num_iter)
    rotated_loadings = varimax_rotation(loadings)

    return rotated_loadings

def cosine_similarity_loss_function(tensor: torch.Tensor, num_iter:int = 100):
    loadings = factor_analysis(tensor, num_iter)
    cosine_similarity = torch.abs(cosine_similarity_matrix(loadings))
    sum_cosine_similarity = torch.sum(cosine_similarity)

    return sum_cosine_similarity