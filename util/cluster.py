import pandas as pd
import numpy as np
from util.utility import *

# Feature의 Information을 추출한 행렬로부터 비슷한 Feature끼리 묶어주는 방법들을 정리합니다.

# Input = Data_Info_Array (maybe it's shape = (n*n); n = # of Features)
# Output = List of Subset features, like [[0, 3], [1, 2], [4, 5, 6]]

class Cluster_Func():
    def __init__(self, info_matrix:np.array):
        self.info_matrix = info_matrix

    def cos_sim(self, threshold:int=0.5):
        cos_sim = array_cos_sim(self.info_matrix)
        under_cos_sim = threshold_below_array(cos_sim, threshold)
        total_result = sim_result(under_cos_sim)
        result, result_idx = filter_sublists(total_result)
        
        unique_list = []
        for sublist in result:
            if sublist not in unique_list:
                unique_list.append(sublist)
        return unique_list