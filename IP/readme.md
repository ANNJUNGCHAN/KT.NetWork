# 시작하기

## 문제의 정의

실시간 IP 할당 개수 추이를 기반으로 이상 발생 시점을 탐지하는 문제

## 데이터 설명

### 전체적인 데이터에 대한 설명

- DHCP 장비 1종으로부터 수집된 10분 주기의 IP 세션 데이터 12개월치가 제공
- 주어진 데이터를 활용하여, 2021년 하반기(7월 1일 - 12월 31일) IP 할당 프로세스의 이상 발생 시점을 예측
- 일부 데이터에 대해선 결측치가 존재

### 데이터 칼럼 설명

| columns | Explain |
| --- | --- |
| Timestamp | [YYYYMMDD_HHmm(a)-HHmm(b)] 형식을 가지며, 수집 범위는 YYYY년 MM월 DD일 HH시 mm분(a) 부터 HH시 mm분(b) |
| Svr_detect | DHCP 프로세스에서 단위 시간 내 클라이언트인 단말들이 DHCP 서버에게 연결을 요청한 횟수 |
| Svr_connect | DHCP 프로세스에서 단위 시간 내 클라이언트인 단말들에게 DHCP 서버와 연결이 확립됨을 나타내는 횟수 |
| Ss_request | DHCP 프로세스에서 단위 시간 내 서버에 연결된 단말들이 IP 할당을 요청한 횟수 |
| Ss_established | IP 할당 요청을 받은 DHCP 서버가 클라이언트에게 IP가 할당됨을 나타내는 횟수 |

## DHCP 장비 설명

- Dynamic Host Configuration Protocol
- 클라이언트(단말)의 요청에 따라 IP 주소를 동적으로 할당 및 관리
- DHCP 서버는 서버와 클라이언트를 중개하는 방식으로 요청 단말에게 IP를 할당

## 제출 형식

- 정상(0)과 이상(1)을 판단하기 위한 모델을 학습
- 모델을 활용해, 2021년 하반기 전체에 대한 예측을 수행
- 제출 파일은 2021년 7월 1일 00시 00분-10분 부터 2021년 12월 31일 23시 50분-00분 구간의 이상 이벤트를 예측한 .csv 형식으로 저장
- 예측 데이터프레임의 크기는 [26496 * 1]

# EDA

## 라이브러리 및 데이터

코랩 환경에서 데이터를 불러오기 위해 google.colab의 drive 사용

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

데이터 전처리를 위해 numpy와 pandas 패키지 사용

```python
import numpy as np
import pandas as pd
```

그래프 시각화를 위해 matplotlib 패키지 사용

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

경고 메시지를 필터링 하기 위해 warning 사용

```python
import warnings
warnings.filterwarnings('ignore')
```

시계열 데이터에서 추세제거를 위해 fbprophet 패키지 사용

```python
!pip install fbprophet
from fbprophet import Prophet
```

데이터 불러오기

```python
data=pd.read_csv("/content/gdrive/MyDrive/KT/data/IP/DHCP.csv")
```

## 데이터 전처리

### 시간 데이터 재구성

“-“를 기준으로 Timestamp열의 시간 데이터가 날짜와 시간으로 분리되기에, 이를 str.split 함수를 사용하여 분리한 후, date열과 time열로 각각 분리시킴.

```python
time=data["Timestamp"].str.split("-", expand = True)
```

date열의 문자열 0번째부터 3번째까지는 년도, 4번째부터 5번째까지는 월, 6번째부터 7번째까지는 일수를 표시한다는 사실을 알 수 있었고, time열의 문자열의 0번째부터 1번째까지는 시간(hour), 2번째부터 3번째까지는 분(minutes), 4번째부터 5번째 까지는 초(second)를 나타낸다는 사실을 알 수 있었음. 그러므로 년도, 월, 일, 시간, 분을 모두 분리하여 리스트에 저장한 후, 이를 numpy의 datatime64의 형태(yyyy-mm-dd hh:mm:ss)로 모두 바꾸어주었음. 이 때, 초는 00으로 모두 일정하므로, 문자열의 맨 뒤에 :00을 붙여주었음.

```python
time.drop(columns = [1], inplace = True)
time=time[0].str.split("_", expand = True)
date = time[[0]]
time = time[[1]]

year = []
for i in range(len(date)) :
    year.append(date[0][i][0:4])
 
month = []
for i in range(len(date)) :
    month.append(date[0][i][4:6])
 
day = []
for i in range(len(date)) :
    day.append(date[0][i][6:8])

hour = []
for i in range(len(date)) :
    hour.append(time[1][i][0:2])
 
min = []
for i in range(len(date)) :
    min.append(time[1][i][2:4])

time_data = pd.DataFrame({'year':year,'month':month,'day':day, 'hour' : hour, 'min':min})
timestamp = time_data['year'] +"-" +time_data['month'] + "-" +time_data['day'] + " " + time_data['hour'] +":" + time_data["min"] + ":" + "00"
data["Timestamp"] = timestamp

data["month"] = data["month"].astype(int)

```

### 훈련데이터와 테스트 데이터 분리

Train 데이터와 Test 데이터는 “데이터 설명”의 “3. 제출방법”에 의해서 7월을 기준으로 Train 데이터와 Test 데이터가 분리된다는 사실을 알 수 있었고, 이 규칙에 따라 7월 이전의 데이터는 Train 데이터로, 7월 이후의 데이터는 Test 데이터로 지정함.

```python
train=data[data["month"]<7]
test=data[data["month"]>=7]
```

### 결측값 보간

시계열 데이터이므로, 결측치들을 시계열 날짜 index 기준으로 결측값을 보간해야 된다고 판단하여 interpolate(method = “time”) 함수를 사용하여 결측값들을 보간함.

```python
train.index = train["Timestamp"].astype("Datetime64")
train=train.drop(columns = ["Timestamp"])
train = train.interpolate(method = "time")
train = train.reset_index()
 
test.index = test["Timestamp"].astype("Datetime64")
test=test.drop(columns = ["Timestamp"])
test = test.interpolate(method = "time")
test = test.reset_index()
```

### 추세 제거

시계열 분해를 실시했을 때, 데이터가 추세를 갖는다는 사실을 알 수 있었음. 오토인코더 모델로 이상을 탐지할 계획이었으므로, 데이터의 추세를 모두 제거하여 오토인코더 모델이 학습하는데 지장이 없도록 함.

```python
# Svr_detect
Svr_detect = pd.concat([train['Timestamp'],train["Svr_detect"]],axis = 1)
Svr_detect = Svr_detect.rename(columns = {'Timestamp' : 'ds', "Svr_detect" : 'y'})
Svr_detect_model = Prophet()
Svr_detect_model.fit(Svr_detect)
Svr_detect_predict = Svr_detect_model.predict(pd.DataFrame(Svr_detect["ds"]))
Svr_detect["trend_off"]=Svr_detect['y'] - Svr_detect_predict['trend']
 
Svr_detect_test = pd.concat([test['Timestamp'],test["Svr_detect"]],axis = 1)
Svr_detect_test = Svr_detect_test.rename(columns = {'Timestamp' : 'ds', "Svr_detect" : 'y'})
Svr_detect_test_model = Prophet()
Svr_detect_test_model.fit(Svr_detect_test)
Svr_detect_test_predict = Svr_detect_test_model.predict(pd.DataFrame(Svr_detect_test["ds"]))
Svr_detect_test["trend_off"]=Svr_detect_test['y'] - Svr_detect_test_predict['trend']

# Svr_connect
Svr_connect = pd.concat([train['Timestamp'],train["Svr_connect"]],axis = 1)
Svr_connect = Svr_connect.rename(columns = {'Timestamp' : 'ds', "Svr_connect" : 'y'})
Svr_connect_model = Prophet()
Svr_connect_model.fit(Svr_connect)
Svr_connect_predict = Svr_connect_model.predict(pd.DataFrame(Svr_connect["ds"]))
Svr_connect["trend_off"]=Svr_connect['y'] - Svr_connect_predict['trend']
 
Svr_connect_test = pd.concat([test['Timestamp'],test["Svr_connect"]],axis = 1)
Svr_connect_test = Svr_connect_test.rename(columns = {'Timestamp' : 'ds', "Svr_connect" : 'y'})
Svr_connect_test_model = Prophet()
Svr_connect_test_model.fit(Svr_connect_test)
Svr_connect_test_predict = Svr_connect_test_model.predict(pd.DataFrame(Svr_connect_test["ds"]))
Svr_connect_test["trend_off"]=Svr_connect_test['y'] - Svr_connect_test_predict['trend']

# Ss_request
Ss_request = pd.concat([train['Timestamp'],train["Ss_request"]],axis = 1)
Ss_request = Ss_request.rename(columns = {'Timestamp' : 'ds', "Ss_request" : 'y'})
Ss_request_model = Prophet()
Ss_request_model.fit(Ss_request)
Ss_request_predict = Ss_request_model.predict(pd.DataFrame(Ss_request["ds"]))
Ss_request["trend_off"]=Ss_request['y'] - Ss_request_predict['trend']
 
Ss_request_test = pd.concat([test['Timestamp'],test["Ss_request"]],axis = 1)
Ss_request_test = Ss_request_test.rename(columns = {'Timestamp' : 'ds', "Ss_request" : 'y'})
Ss_request_test_model = Prophet()
Ss_request_test_model.fit(Ss_request_test)
Ss_request_test_predict = Ss_request_test_model.predict(pd.DataFrame(Ss_request_test["ds"]))
Ss_request_test["trend_off"]=Ss_request_test['y'] - Ss_request_test_predict['trend']

# Ss_Established
Ss_Established = pd.concat([train['Timestamp'],train["Ss_Established"]],axis = 1)
Ss_Established = Ss_Established.rename(columns = {'Timestamp' : 'ds', "Ss_Established" : 'y'})
Ss_Established_model = Prophet()
Ss_Established_model.fit(Ss_Established)
Ss_Established_predict = Ss_Established_model.predict(pd.DataFrame(Ss_Established["ds"]))
Ss_Established["trend_off"]=Ss_Established['y'] - Ss_Established_predict['trend']
 
Ss_Established_test = pd.concat([test['Timestamp'],test["Ss_Established"]],axis = 1)
Ss_Established_test = Ss_Established_test.rename(columns = {'Timestamp' : 'ds', "Ss_Established" : 'y'})
 
Ss_Established_test_model = Prophet()
Ss_Established_test_model.fit(Ss_Established_test)
Ss_Established_test_predict = Ss_Established_test_model.predict(pd.DataFrame(Ss_Established_test["ds"]))
Ss_Established_test["trend_off"]=Ss_Established_test['y'] - Ss_Established_test_predict['trend']
```

#### Train 데이터의 Svr_detect열의 시계열 분해
![Untitled](https://user-images.githubusercontent.com/89781598/193590042-c14dbb75-eced-4a88-aa8d-aef92c6e405d.png)

#### Test 데이터의 Svr_detect열의 시계열 분해
![Untitled 1](https://user-images.githubusercontent.com/89781598/193590093-fa53b8de-7cc4-460a-ab96-ee0e114eceb9.png)

#### Train 데이터의 Svr_connect열의 시계열 분해
![Untitled 2](https://user-images.githubusercontent.com/89781598/193590120-fe7c1a8c-f782-4768-bd1b-11aa2040ebe5.png)

#### Test 데이터의 Svr_connect열의 시계열 분해
![Untitled 3](https://user-images.githubusercontent.com/89781598/193590143-bceaa4ad-faf2-4e37-ad69-1afc987407e3.png)

#### Train 데이터의 Ss_request열의 시계열 분해
![Untitled 4](https://user-images.githubusercontent.com/89781598/193590171-7efa23b7-5808-4d21-812a-651d52e81af3.png)

#### Test 데이터의 Ss_request열의 시계열 분해
![Untitled 5](https://user-images.githubusercontent.com/89781598/193590190-35d8ceb9-deea-4458-9981-222256cd115e.png)


#### Train 데이터의 Ss_Established열의 시계열 분해
![Untitled 6](https://user-images.githubusercontent.com/89781598/193590216-edf5838c-e409-407c-9955-b84ac440a751.png)

#### Test 데이터의 Ss_Established열의 시계열 분해
![Untitled 7](https://user-images.githubusercontent.com/89781598/193590236-bb66359d-d3a5-4650-92bc-9163300761cb.png)

#### 시계열 분해로써 얻은 정보

- 모든 열을 시계열 분해하였을 때, 모든 열에서 추세의 값이 매우 큰 폭으로 불규칙하게 변화하는 것을 알 수 있다.
- 또한, 모든열은 어느정도의 주기성을 가지고 있는 것이 판단된다.
- 오토인코더 모델을 사용하여 이상치를 탐지할 때, 최대한 추세와 주기성에 대한 정보가 없어야 오토인코더 모델이 이상치를 잘 탐지할 것이다.
- 주기성의 경우 배치를 설정할 때, 데이터를 랜덤하게 섞어서 주기성을 띄지 않게 할 수 있지만, 추세의 경우, 큰 폭으로 불규칙하게 변하기 때문에 전처리과정에서 추세를 제거하였다.

### 추세 제거 데이터로의 대체

오토인코더 모델은 input 데이터를 인코더를 통해 압축한 후, decoder를 사용하여 input 데이터를 재구현하는 모델임. 이 때, 재구현한 데이터와 input 데이터 간의 차이가 크면 이상치라고 판정함. 이 때, 데이터에 추세가 존재하여 데이터의 값이 바뀐다면, 오토인코더는 이를 이상치라고 판정할 위험이 있음. 그러므로, 데이터의 추세를 제거함. 이 때, Train데이터와 Test 데이터의 추세를 각각 따로 제거하였으며, 모든 열에 대해서 모두 추세를 제거하였음. 추세를 제거할 때, Prophet() 모델을 fitting시켜서 추세만 추출하여 원래의 값(y)에서 추세(trend)를 제거하였음. 그 후, 기존의 데이터를 추세가 제거된 데이터로 대체하였음.

```python
train['Svr_detect'] = Svr_detect['trend_off']
train['Svr_connect'] = Svr_connect['trend_off']
train['Ss_request'] = Ss_request['trend_off']
train['Ss_Established'] = Ss_Established['trend_off']
 
test['Svr_detect'] = Svr_detect_test['trend_off']
test['Svr_connect'] = Svr_connect_test['trend_off']
test['Ss_request'] = Ss_request_test['trend_off']
test['Ss_Established'] = Ss_Established_test['trend_off']
```

### 스케일링

모델이 모든 열의 데이터를 동일한 비중으로 인식하도록 sklearn패키지의 preprocessing에 있는 MinMaxScaler를 이용하여 MinMaxScaling을 진행함. 이 때, 스케일링을 train, test 각각 적용시킴.

```python
train.drop(columns = ["Timestamp","month"],inplace = True)
test.drop(columns = ["Timestamp","month"],inplace = True)
 
data = pd.concat([train,test])
data = data.reset_index()
data.drop(columns = ["index"],inplace = True)
 
from sklearn.preprocessing import MinMaxScaler
model = MinMaxScaler()
model = model.fit(data)
 
train = model.transform(train)
test = model.transform(test)
```

### 텐서 변환

파이토치에서 딥러닝 모델이 데이터를 인식하도록 하기 위해서, torch.Tensor를 사용하여 데이터를 텐서 형태로 변환시킴.

```python
train_data = torch.Tensor(np.array(train))
test_data = torch.Tensor(np.array(test))
```
# 모델링

## 라이브러리

오토인코더 사용을 위한 torch 사용

```python
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
```

랜덤 시드 구성

```python
random_seed = 1339
SEED = 1339
torch.manual_seed(random_seed)
np.random.seed(random_seed)
import random
import os
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
 
set_seeds()
```

## 모델 선택

오토인코더란, 단순히 입력을 출력으로 복사하는 간단한 신경망이다. 주로 이미지를 생성하는데 쓰이는 신경망이지만, 최근 이상탐지에 대한 연구논문에서는 이러한 오토인코더를 이상탐지에 많이 사용하고 있다.

 오토인코더는 네트워크에 여러가지 방법으로 제약을 줌으로써 어려운 신경망으로 만든다. 예를들어  hidden layer의 뉴런 수를 input layer(입력층) 보다 작게해서 데이터를 압축(차원을 축소)한다거나, 입력 데이터에 노이즈(noise)를 추가한 후 원본 입력을 복원할 수 있도록 네트워크를 학습시키는 등 다양한 오토인코더가 있다. 이러한 제약들은 오토인코더가 단순히 입력을 바로 출력으로 복사하지 못하도록 방지하며, 데이터를 효율적으로 표현(representation)하는 방법을 학습하도록 제어한다.

![Untitled](https://user-images.githubusercontent.com/89781598/193590875-fec24a73-796a-4e3f-bc1b-228887fbf230.png)


이 때, 가장 기본적인 오토인코더는 1차원 데이터가 input으로 들어가는 오토인코더이다.

![Untitled 1](https://user-images.githubusercontent.com/89781598/193590938-5c914880-778c-4ba5-b445-57cff8536acd.png)


이 때, 1차원 데이터란, 하나의 행으로 이루어진 데이터로써, 한 단위의 시간만 고려한다.

이러한 한단위에 해당하는 데이터를 오토인코더의 인코더를 통해 축소시키고, 디코딩으로 다시 원래상태로 돌렸을 때, 둘 간의 차이가 크면 loss가 큰 것이고, 둘 간의 차이가 작으면 loss가 작게 나타날 것이다. loss란 얼마나 오토인코더가 기존의 데이터를 그대로 구현해내는지를 알려주며, loss값이 작다면 오토인코더가 기존의 데이터를 잘 구현해내는 것이고, loss값이 크다면, 오토인코더가 기존의 데이터를 그대로 구현해내지 않는다는 것을 의미한다.

우리가 오토인코더를 학습시킬 때, 정상인 데이터를 가지고 학습을 시키므로, 모델은 정상인 데이터만 학습할 것이다. 그러므로, 정상인 데이터만 학습한 오토인코더는 정상인 데이터가 input으로 들어왔을 때 output을 input과 비슷하게 재현시킬 것이고, 비정상 데이터의 경우 input으로 들어온다고 하더라도 output의 값이 재현이 안될 것이다. 오토인코더는 output의 값에 대해서 학습한 적이 없기 때문이다. 이러한 원리를 이용하여, 오토인코더를 이용하여 이상을 탐지해보려고 한다. 

## 변수 선택

변수는 4개의 변수가 모두 중요하다고 판단하여 4개의 변수를 모두 사용하였으며, 우리가 넣을 변수는 Svr_detect, Svr_connect, Ss_request, Ss_Established임.

## 모델 구축

모델은 아래와 같이 구축하였음.

![Untitled 2](https://user-images.githubusercontent.com/89781598/193590965-3d266b41-c6a6-44b8-a900-3cddf8043113.png)

변수의 개수가 적고, 오토인코더가 완전히 학습되면 output값을 input값과 거의 동일하게 재현하기 때문에, 오토인코더 안의 레이어를 최대한 단순하게 구성하기 위해서 완전연결망(Linear)을 사용하였고, 활성화함수는 RReLU 함수를 사용함. 또한, 인코더에서 4개의 변수로 이루어진 데이터(4차원)를 완전연결망을 이용하여 3차원으로 줄인 후, RReLU함수를 적용하여 데이터의 값을 변환한 후, 다시 이를 완전연결망을 사용하여 3차원 데이터를 2차원으로 축소시키고 RReLU함수를 적용함. 마지막으로, 디코더에서는 이와 반대되는 순서로 완전연결망을 사용하여 2차원 데이터를 3차원 데이터로 늘린 후, RReLU함수를 적용한 후, 다시 완전연결망을 사용하여 3차원 데이터를 4차원 데이터로 늘림.

코드는 아래와 같음.

```python
class Model(nn.Module) :
    def __init__(self, input_size, hidden_size, output_size) :
        super(Model, self).__init__()
 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
 
        # 인코더
        self.Encoder = nn.Sequential(
        
            nn.Linear(input_size,hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0],output_size),
            nn.RReLU()
        )
 
        # 디코더
        self.Decoder = nn.Sequential(
 
            nn.Linear(output_size,hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0],input_size)
        )
 
    # 전방 전달
    def forward(self, x) :
        x = self.Encoder(x)
        x = self.Decoder(x)
 
        return x
 
# input_size, hidden_size, output_size 결정
input_size = 4
hidden_size = [3]
output_size = 2
 
# 모델 최종 구성
model = Model(input_size, hidden_size, output_size)
print(list(model.modules()))
```

## 모델의 학습과 검증

### 학습함수 구성

- 옵티마이저는 Adam을 사용하였고, learning rate를 0.01로 설정함.
- 파이토치의 DataLoader를 사용하여 배치 사이즈를 64로 설정하고, 데이터가 뒤섞이도록 하기위해서 shuffle = True로 설정함.
- 데이터가 뒤섞이도록 한 이유는 데이터를 random하게 추출하여 모델이 편향되지 않은 결과를 나오도록 하기 위해서임. 이 때, 이미 앞에서 데이터의 추세를 제거하였으므로, 시간순서가 섞여도 상관이 없다고 판단함.(추세를 제거함으로써 시간에 따른 증가, 감소를 고려하지 않아도 되며, 임의로 뒤섞었기 때문에 오히려 주기성이 사라져서 시계열은 정상성을 띄게 된다고 판단함.)
- 너무 모델이 학습을 잘하면 이상치를 잡아낼 수 없다고 판단함.(오토인코더는 input 데이터를 최대한 재현하려고 하기 때문에, 모델이 학습을 잘하게 된다면 input 데이터와 output 데이터의 차이가 거의 나지 않아 이상치를 탐색할 수 없게 되어버림.) 그러므로, 에포크를 10으로 설정함.
- input 데이터와 output 데이터의 차이를 손실함수를 MSE로 설정하여 구함.

```python
# 손실함수를 MSE 함수를 사용

loss_function = nn.MSELoss()

def training(model, data,loss_function):

    # 옵티마이져는 Adam을 사용, learning rate는 0.01로 설정

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    

    # 데이터를 불러올 때, 배치사이즈는 64로 주고, shuffle = True로 설정하여 데이터를 뒤섞음
    dataloader = DataLoader(data, batch_size =64, shuffle =True)
    
    for epoch in range(1, 11): # 에포크는 10번 줌
        
        update_loss = 0.0
        
        for x in dataloader :
            optimizer.zero_grad()
            output = model(x)
            
            loss=loss_function(x, output) #dataloader로 불러온 데이터 값과 실제 데이터 간의 MSE 산출
            loss.backward()
            
            optimizer.step()
            update_loss += loss.item()

        print('epoch:', f'{epoch}', '  loss:', f'{update_loss}')
    
    return model
```

### 학습

아래의 코드로 모델을 학습시킴.

```python
# 모델 학습
Model = training(model, train_data, loss_function)

# 모델 저장
torch.save(Model.state_dict(),'/content/gdrive/MyDrive/model.pt')
```

### 손실 값 구하기

아래의 코드로 손실 값을 구함.

```python
# train 데이터의 손실값 구하기
train_loss = []
for data in train_data :
  output = Model(data)
  loss = loss_function(output, data)
  train_loss.append(loss.item())

# test 데이터의 손실값 구하기
test_loss = []
for data in test_data :
  output = Model(data)
  loss = loss_function(output, data)
  test_loss.append(loss.item())
```

### 임계 값 설정

이 때, 0.0016을 가장 이상적인 임계값이라고 판단하였으나, Train 데이터의 이상치 개수와 Test 데이터의 이상치 개수가 현실적으로 같을 수는 없다고 판단함. 이에, 임계 값을 0.0016부터 0.0020까지와 0.0013부터 0.0020까지 제출해보면서 결과를 확인함. 그 결과, 최적의 이상치가 0.00131이라고 판단하였음. 

아래의 코드로 그래프를 생성함.

```python
# train 데이터와 test 데이터의 손실값의 갯수를 비교하기
train_len = []
test_len = []
index = np.arange(0.001, 0.002, 0.00001)
for i in np.arange(0.001, 0.002, 0.00001) :
    train_len_error = list(train_loss >= i)
    test_len_error = list(test_loss >= i)
    train_len.append(train_len_error.count(True))
    test_len.append(test_len_error.count(True))

plt.figure(figsize = (10,6))
plt.plot(index, train_len)
plt.plot(index, test_len)
```

![Untitled 3](https://user-images.githubusercontent.com/89781598/193591010-ef203f26-ca39-4466-992e-a520aeeacece.png)

이에 아래의 코드로 임계값을 설정함.

```python
# 임계값을 설정하기

treshold = np.array(0.00131)
train_error = list(train_loss >= treshold)
test_error = list(test_loss >= treshold)

print("train_error : ",train_error.count(True))
print("test_error : ",test_error.count(True))

print("Threshold : ",treshold)
```

또한, 아래와 같이 임계값을 확대함. 그 이유는 임계 값을 설정하고, 어느 지점에서 오류가 났는지 확인해본 결과, 오류가 연속적으로 일어난다는 사실을 알 수 있었음. 이러한 사실을 바탕으로, 어떠한 이상치가 오토인코더로 감지되었다면, 이후 한 개의 시점에서 확정적으로 이상치가 존재할 것이라는 조건을 추가시킴. 이러한 조건을 추가시킨 후, 모델의 성능이 더욱 개선됨을 확인함.

```python
###################### false가 난 구간 확인 #######################
submit = []
for i in range(len(test_loss)) :
    if test_loss[i] >= treshold :
        submit.append(1)
    else :
        submit.append(0)

submit = pd.DataFrame(submit)
submit=submit.rename(columns = {0:'Prediction'})

false = []
for i in range(len(submit)) :
    if submit["Prediction"][i] == 1:
         false.append(i)

false = list(false)

######### 오토인코더가 오류를 감지한 이후 1 시점에서 ################
######### 오류가 확정적으로 일어난다는 조건을 넣음 ##################
many = 1
false_plus = []
for i in range(len(false)) :
    false_plus.append(false[i] + many)

false_list = set(false) | set(false_plus)
false_list = list(false_list)
false_list.sort()

#################### 제출 ############################
submit_many = pd.DataFrame()
submit_many["Prediction"] = np.zeros(26496)

for i in false :
    submit_many["Prediction"][i] = 1

for i in false_list :
    submit_many["Prediction"][i] = 1

submit_many["Prediction"].value_counts()
```

### 결과 도출

아래의 코드로 결과를 도출함.

```python
submit_many.to_csv("/content/gdrive/MyDrive/part1_result.csv")
```

## 결론 및 제언

- 오토인코더는 생성모델로써, input의 값을 정확하게 재현하는 것을 목표로 하는 모델이다. 우리는 이를 역이용하여 최대한 오토인코더를 단순하게 적합시켜 input값을 정확하게 재현하지 못하도록 하였다. 정확하게 재현을 못하도록 만든 이유는 이상치가 오토인코더의 input으로 들어왔을 때, 오토인코더의 input값과 이러한 input값을 재현한 값인 ouput의 값의 차이가 커져서 input과 output간의 차이가 큰 관측치를 이상치로 판별할 수 있기 때문이다.
- 우리는 임계 값을 기준으로 임계값을 넘어가는 관측치를 이상치로 보고, 임계 값보다 작은 관측치를 정상으로 보았다. 하지만, 임계 값을 정할 때, Train 데이터의 이상치 개수와 Test 데이터의 이상치 개수가 최대한 동일한 임계값을 기준으로 grid search를 진행하여 최적의 임계값을 찾아내었다. 이러한 grid search로 최적의 임계값을 찾아내는 것은 시간적인 한계가 존재한다. 이러한 단점을 보완하기 위해서 만약 과거에 일어났던 관측치에 이상치 여부를 판정한 라벨이 있다면, Train 데이터의 이상치를 모두 제거한 후, 정상인 데이터만 가지고 오토인코더를 학습시켰을 때 더 이상치를 잘 잡아낼 수 있다고 생각한다.
