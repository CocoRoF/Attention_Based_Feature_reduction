import pandas as pd
from sklearn.decomposition import PCA
# PCA를 사용하여 유사한 열들을 통합하는 함수
def integrate_columns_with_pca_v2(data, group_indices):
    integrated_data = []
    column_names = []
    
    # DataFrame에서 사용할 열 이름 생성
    for group in group_indices:
        if len(group) > 1:
            column_names.append('_'.join(map(str, group)))
        else:
            column_names.append(str(group[0]))
    
    for group in group_indices:
        if len(group) > 1:  # 유사한 그룹으로 묶인 열
            pca = PCA(n_components=1)
            group_data = np.array(data)[:, group]
            transformed_data = pca.fit_transform(group_data)
            integrated_data.append(transformed_data.flatten())
        else:  # 다른 그룹과 유사하지 않아 단독으로 존재하는 열
            integrated_data.append(np.array(data)[:, group[0]])

    integrated_df = pd.DataFrame(np.array(integrated_data).T, columns=column_names)

    return integrated_df

# 데이터를 PCA를 통해 열 통합
# pca_integrated_df = integrate_columns_with_pca_v2(data, fa.result)

# pca_integrated_df
