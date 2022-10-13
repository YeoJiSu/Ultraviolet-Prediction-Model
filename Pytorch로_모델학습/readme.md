# 코드를 돌리는 위치에 models 폴더 생성하여 생성된 모델 파일이 저장되도록 한다.

# 1.py

- epoch = 10000 -> 200 마다 확인
- hidden_size = 2 -> hidden state 개수
- activation 함수 = relu
- loss 함수 = mse
- optimizer = adam
- learning_rate = 0.001
- num_layer = 1 -> lstm 계층

# 2.py 

- 데이터셋을 텐서 형태로 정의하는 부분  -> TensorDataset,DataLoader 사용(여기서 배치사이즈 설정 가능함)
- epoch = 1000 -> 20 마다 확인
- layer가 fully connected 방식
- loss 함수 = mse
- optimizer = adam
- hidden_size = 10 
- learning_rate = 0.01 # 학습률
- 학습시 earlyStop 추가

# 3.py

- data shuffle = False
- self.fc 층 더 늘림
- 활성화 함수 relu 사용
- loss function rmse로 -> mse 한 값에 루트 씌움
- epoch = 2000

