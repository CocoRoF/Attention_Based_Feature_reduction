import torch

# Model에 의한 Error Function만을 정의합니다. 혹은 이에 필요한 함수만 정의합니다.

def pearson_correlation_coefficient(tensor):
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor should be 2D.")
    
    n_samples, n_features = tensor.shape
    mean = torch.mean(tensor, dim=0)
    t_centered = tensor - mean
    denominator = torch.sqrt(torch.sum(t_centered ** 2, dim=0))
    corr_matrix = torch.empty((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            corr_matrix[i, j] = torch.sum(t_centered[:, i] * t_centered[:, j]) / (denominator[i] * denominator[j])
    
    return corr_matrix

def pearson_correlation_coefficient_loss_function(tensor: torch.Tensor):
    pcc = torch.abs(pearson_correlation_coefficient(tensor))
    sum_pcc = torch.sum(pcc)
    return sum_pcc