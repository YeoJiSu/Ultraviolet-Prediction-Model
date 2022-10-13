import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch
import torch.nn as nn
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
train_x_tensor = Variable(torch.Tensor(train_x))
train_y_tensor = Variable(torch.Tensor(train_y))
print("After torch variable shape_Train : ",train_x_tensor.shape, train_y.shape)

test_x_tensor = Variable(torch.Tensor(test_x))
test_y_tensor = Variable(torch.Tensor(test_y))
print("After torch Variable shape_Test : ",test_x_tensor.shape, test_y_tensor.shape)

train_x_tensor_final = torch.reshape(train_x_tensor, (train_x_tensor.shape[0], 1, train_x_tensor.shape[1]))
train_y_tensor_final = torch.reshape(train_y_tensor, (train_y_tensor.shape[0], 1, train_y_tensor.shape[1]))
test_x_tensor_final = torch.reshape(test_x_tensor, (test_x_tensor.shape[0], 1, test_x_tensor.shape[1]))
test_y_tensor_final = torch.reshape(test_y_tensor,(test_y_tensor.shape[0], 1, test_y_tensor.shape[1]) )
print(train_x_tensor_final.shape, test_x_tensor_final.shape)

# GPU setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# LSTM network modeling
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) : 
        super(LSTM, self).__init__()
        self.num_classes = num_classes # 클래스 개수
        self.num_layers = num_layers # lstm 계층의 개수
        self.input_size = input_size # 입력 크기 
        self.hidden_size = hidden_size # 은닉층 뉴런 개수
        self.seq_length = seq_length # 시퀀스 길이 
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True) # LSTM 계층
        self.layer_1 = nn.Linear(hidden_size, 256) # 연결층
        self.layer_2 = nn.Linear(256,256)
        self.layer_3 = nn.Linear(256,128)
        self.layer_out = nn.Linear(128, num_classes) # 출력층
        self.relu = nn.ReLU() #Activation Func

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉층 상태를 0 으로 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #셀 상태를 0으로 초기화

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # LSTM 계층에 은닉 상태와 셀 상태 적용
        
        hn = hn.view(-1, self.hidden_size) # 연결층 적용을 위해 데이터의 형태 조정
        out = self.relu(hn) #pre-processing for first layer
        out = self.layer_1(out) # first layer
        out = self.relu(out) # activation func relu
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.layer_3(out)
        out = self.relu(out)
        out = self.layer_out(out) #Output layer

        return out
    
# Code Main
num_epochs = 10000
learning_rate = 0.001
input_size = int(len(scale_cols)) # input size 
hidden_size = 2 # 은닉층의 뉴런/유닛 개수
num_layers = 1 # LSTM 계층의 개수 -> 이거 2,3 늘리면 target size 개수 변형 일나는디 ,,,? Using a target size (torch.Size([708300, 1])) that is different to the input size (torch.Size([2124900, 1])) 요런식으로..쩝..
num_classes = 1 # output size 인 듯 .?!
 
model= LSTM(num_classes, input_size, hidden_size, num_layers, train_x_tensor_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in tqdm(range(num_epochs)): 
    outputs = model.forward(train_x_tensor_final.to(device))
    optimizer.zero_grad()
    loss = loss_function(outputs, train_y_tensor.to(device))
    loss.backward()
    optimizer.step() # improve from loss = back propagation -> 오차 업데이트
    if epoch % 200 == 0 :
        print("Epoch : %d, loss : %1.5f" % (epoch, loss.item()))

# 모델 저장하기
PATH = './models/'
torch.save(model, PATH + '1.pt')  # 전체 모델 저장
torch.save(model.state_dict(), PATH + '1_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + '1.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능

# Estimated Value
test_predict = LSTM(test_x_tensor_final.to(device)) #Forward Pass
predict_data = test_predict.data.detach().cpu().numpy() #numpy conversion
predict_data = scaler.inverse_transform(predict_data) #inverse normalization(Min/Max)

# Real Value
real_data = test_y_tensor.data.numpy() # Real value
real_data = scaler.inverse_transform(real_data) #inverse normalization

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
import matplotlib.pyplot as plt
plt.figure(figsize = (30,6)) # Plotting
plt.plot(real_data, 'b', label = 'Real Data')
plt.legend()
plt.show()

plt.figure(figsize=(30, 6))
plt.plot(predict_data, 'r', label = 'prediction')
plt.legend()
plt.show()