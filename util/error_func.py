import torch

def pearson_correlation_coefficient(tensor):
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor should be 2D.")
    n_samples, n_features = tensor.shape
    # 평균을 계산합니다.
    mean = torch.mean(tensor, dim=0)
    # 각 열에서 평균을 뺀 값을 계산합니다.
    t_centered = tensor - mean
    # 분모의 값을 계산하기 위한 각 열의 제곱 합을 계산합니다.
    denominator = torch.sqrt(torch.sum(t_centered ** 2, dim=0))
    # 상관계수 행렬을 초기화합니다.
    corr_matrix = torch.empty((n_features, n_features))
    # 상관계수 행렬의 각 요소를 계산합니다.
    for i in range(n_features):
        for j in range(n_features):
            corr_matrix[i, j] = torch.sum(t_centered[:, i] * t_centered[:, j]) / (denominator[i] * denominator[j])
    
    return corr_matrix

def pearson_correlation_coefficient_loss_function(tensor: torch.Tensor):
    pcc = torch.abs(pearson_correlation_coefficient(tensor))
    sum_pcc = torch.sum(pcc)
    return sum_pcc