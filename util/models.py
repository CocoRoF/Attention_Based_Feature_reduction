import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#

class feature_combination():
    def __init__(self, raw_data_matrix:np.array, information_matrix:np.array, similarity_feature_list:list):
        self.raw_matrix = raw_data_matrix
        self.information_matrix = information_matrix
        self.similarity_feature_list = similarity_feature_list
        
        self.selected_array_list = []
        for i in self.similarity_feature_list:
            selected_array = self.raw_matrix[:, i]
            self.selected_array_list.append(selected_array)
        self.selected_raw_tensor_list = [torch.Tensor(arr).double().to(device) for arr in self.selected_array_list]
        
        self.selected_info_matrix_list = []
        for i in self.similarity_feature_list:
            selected_value = self.information_matrix[i, :]
            self.selected_info_matrix_list.append(selected_value)
        self.selected_information_tensor_list = [torch.Tensor(arr).double().to(device) for arr in self.selected_info_matrix_list]
        
    def subset_data(self):
        return self.selected_raw_tensor_list, self.selected_information_tensor_list
    
class Attention_512(nn.Module):
    def __init__(self, n_factor:int):
        super(Attention_512, self).__init__()
        self.attention_query = nn.Parameter(torch.randn(n_factor, 512).double().to(device), requires_grad=True)
        self.attention_key = nn.Parameter(torch.randn(n_factor, 512).double().to(device), requires_grad=True)
        self.attention_value = nn.Parameter(torch.randn(n_factor, 512).double().to(device), requires_grad=True)
        self.output_linear_1 = nn.Linear(512, 256).to(device)
        self.output_linear_2 = nn.Linear(256, 128).to(device)
        self.output_linear_3 = nn.Linear(128, 64).to(device)
        self.output_linear_4 = nn.Linear(64, 1).to(device)
            
    def forward(self, selected_raw_tensor_list:list, selected_information_tensor_list:list, sample_num:int):
        input_1 = selected_raw_tensor_list
        input_2 = selected_information_tensor_list
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
        
        self.total_result = torch.empty(sample_num, 0).to(device)
        for num in range(len(self.score_weight_list)):
            result = torch.matmul((input_1[num]).to(device), self.score_weight_list[num].to(device)).to(device)
            self.total_result = torch.cat((self.total_result, result), dim=1).to(device)
        
        return self.total_result
        
    
        
        