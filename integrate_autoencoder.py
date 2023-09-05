import torch
import torch.nn as nn
import torch.optim as optim

# Autoencoder 모델 정의
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, 1)
        self.decoder = nn.Linear(1, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Autoencoder를 사용하여 유사한 열들을 통합하는 함수
def integrate_with_autoencoder_pytorch_v2(data, group_indices):
    ae_transformed_data = []
    column_names = []

    # DataFrame에서 사용할 열 이름 생성
    for group in group_indices:
        if len(group) > 1:
            column_names.append('_'.join(map(str, group)))
        else:
            column_names.append(str(group[0]))

    for group in group_indices:
        if len(group) > 1:  # 유사한 그룹으로 묶인 열
            group_data = np.array(data)[:, group]
            input_dim = group_data.shape[1]
        
            # 데이터 정규화
            mean = group_data.mean()
            std = group_data.std()
            group_data_normalized = (group_data - mean) / std
        
            # 데이터를 텐서로 변환
            group_tensor = torch.tensor(group_data_normalized, dtype=torch.float32)
        
            # Autoencoder 학습
            model = Autoencoder(input_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(5000):
                optimizer.zero_grad()
                outputs = model(group_tensor)
                loss = criterion(outputs, group_tensor)
                loss.backward()
                optimizer.step()
        
            # 인코더를 사용하여 저차원 데이터 추출
            with torch.no_grad():
                encoded_data = model.encoder(group_tensor).numpy()
                ae_transformed_data.append(encoded_data.flatten())
        else:  # 다른 그룹과 유사하지 않아 단독으로 존재하는 열
            ae_transformed_data.append(np.array(data)[:, group[0]])

    ae_integrated_df = pd.DataFrame(np.array(ae_transformed_data).T, columns=column_names)

    return ae_integrated_df

# Autoencoders를 사용하여 유사한 열들을 통합
# ae_integrated_df = integrate_with_autoencoder_pytorch_v2(data, fa.result)

# ae_integrated_df