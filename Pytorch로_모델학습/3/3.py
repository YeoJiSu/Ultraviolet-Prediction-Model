import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
from tqdm import tqdm

# Load datasets
df_all = pd.read_csv("S:/부산대/공모전/2022-06 날씨데이터/날씨 빅데이터 콘테스트/1_1/merge_datasets_without_zero.csv", encoding = 'euc_kr')
df_all = df_all.sort_values(by=['stn','yyyymmdd', 'hhnn'], ascending = [True,True,True])

# test 데이터 가져오기
df = pd.read_csv("S:/부산대/공모전/2022-06 날씨데이터/날씨 빅데이터 콘테스트/1_1/Validation Data/201908_uv_without_zero.csv", encoding = 'euc_kr')
df = df.sort_values(by=['stn','yyyymmdd', 'hhnn'], ascending = [True,True,True])

# 날짜,시간,statin,'lon','lat',고도,landtype 는 뺐음 - 위경도 넣음
scale_cols = ['band1', 'band2', 'band3','band4','band5',
              'band6','band7','band8','band9','band10','band11','band12',
              'band13','band14','band15','band16','solarza','sateze','esr']

# 데이터 정규화 
scaler = MinMaxScaler()

train_x = scaler.fit_transform(df_all[scale_cols])
train_y = scaler.fit_transform(df_all[['uv']])

test_x = scaler.fit_transform(df[scale_cols])
test_y = scaler.fit_transform(df[['uv']])


# reshape하기
train_x_tensor = torch.FloatTensor(train_x)
train_y_tensor = torch.FloatTensor(train_y)
test_x_tensor = torch.FloatTensor(test_x)
test_y_tensor = torch.FloatTensor(test_y)

# 텐서 형태로 데이터 정의
dataset = TensorDataset(train_x_tensor, train_y_tensor)

batch = 100
# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle=False,  
                        drop_last=True)

# GPU setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 설정값
data_dim = len(scale_cols) # 입력 칼럼 
hidden_size = 10 
output_dim = 1 # 출력 칼럼
learning_rate = 0.001 # 학습률
epoch = 2000
seq_length = 1

class LSTM(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc_1 =  nn.Linear(hidden_dim, 256) #fully connected 1
        self.fc_2 = nn.Linear(256,256)
        self.fc_3 = nn.Linear(256,128)
        self.fc = nn.Linear(128, output_dim, bias = True) # 출력층
        self.relu = nn.ReLU() 
       
        
    # 학습 초기화를 위한 함수 -> 학습시 이전 seq의 영향을 받지 않게 하기 위함
    def reset_hidden_state(self): 
        self.hidden = (
                Variable(torch.zeros(self.layers, self.seq_len, self.hidden_dim)).to(device),
                Variable(torch.zeros(self.layers, self.seq_len, self.hidden_dim)).to(device))
    
    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        out = self.relu(x)
        out = self.fc_1(out) #first Dense
        out = self.relu(out)
        out = self.fc_2(out) #Second Dense
        out = self.relu(out)
        out = self.fc_3(out) #Third Dense
        out = self.relu(out)
        out = self.fc(out) #Final Output
        return out

# 모델 학습 함수
def train_model(model, train_df, num_epochs = None, lr = None, verbose = 20, patience = 20):
     
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    epoch = num_epochs
    
    # epoch마다 loss 저장
    train_hist = np.zeros(epoch)

    for epoch in tqdm(range(epoch)):
        avg_cost = 0
        total_batch = len(train_df)
        
        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples
            # seq별 hidden state reset
            model.reset_hidden_state()
            # H(x) 계산
            outputs = model(x_train.to(device))
            # cost 계산, sqrt 씌워서 RMSE로 게산
            loss = torch.sqrt(loss_function(outputs, y_train.to(device)))  
            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss/total_batch
        train_hist[epoch] = avg_cost        
        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):
            # loss가 커졌다면 early stop
            if train_hist[epoch-patience] < train_hist[epoch]:
                print('\n Early Stopping')
                break
    return model.eval(), train_hist
# 모델 학습
net = LSTM(data_dim, hidden_size, seq_length, output_dim, 1).to(device)  
model, train_hist = train_model(net, dataloader, num_epochs = epoch, lr = learning_rate, verbose = 20, patience = 20)
 
# 모델 저장하기
PATH = './models/'
torch.save(model, PATH + '1.pt')  # 전체 모델 저장
torch.save(model.state_dict(), PATH + '1_state_dict.pt')  # 모델 객체의 state_dict 저장

# 예측 테스트
with torch.no_grad(): 
    pred = []
    for pr in range(len(test_x_tensor)):

        model.reset_hidden_state()

        predicted = model(torch.unsqueeze(test_x_tensor.to(device)[pr], 0))
        predicted = torch.flatten(predicted).item()
        pred.append(predicted)

    # INVERSE
    predict_data = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
    real_data = scaler.inverse_transform(test_y_tensor)


# 성능 평가하기
DIFF = 0.2 # 오차범위
predict_value = pd.DataFrame(predict_data)
real_value = pd.DataFrame(real_data)
result = pd.concat([real_value, predict_value],axis = 1)
result.columns = ['real','predict']
count = 0
for i in range(len(result)):
  if abs(result['real'][i] - result['predict'][i]) <= DIFF:
    count+=1
TP = round(count/len(result),3)
print("오차 범위",DIFF,"일 때 TP 값:",TP)


#Figure
plt.figure(figsize = (30,6)) # Plotting
plt.plot(real_data, 'b', label = 'Real Data')
plt.legend()
plt.show()

plt.figure(figsize=(30, 6))
plt.plot(predict_data, 'r', label = 'prediction')
plt.legend()
plt.show()

plt.figure(figsize=(30, 6))
plt.title('두개 겹쳐놓은 것')
plt.plot(real_data, 'b', label = 'Real Data')
plt.plot(predict_data, 'r', label = 'prediction')
plt.legend()
plt.show()