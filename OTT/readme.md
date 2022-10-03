# 시작하기

## 문제의 정의

실시간 OTT 서비스 이용자 수 추이를 기반으로 이상 발생 시점을 탐지하는 문제

## 데이터 설명

![슬라이드4](https://user-images.githubusercontent.com/89781598/193592770-0f9fe133-e623-4e24-bff2-9845587436af.JPG)

### 전체적인 데이터에 대한 설명

- 네 가지 기능별 서버 13종으로부터 수집된 5분 주기의 세션 데이터 24개월치가 제공
- 주어진 데이터를 활용하여, 2018년 전체(1월 1일 - 12월 31일) Media 데이터의 이상 발생 시점을 예측
- 일부 데이터에 대해선 결측치가 존재

### 데이터 파일 설명

| Data | Explain |
| --- | --- |
| Media_INFO.csv | 상품 가입/해지, 약관 동의, 구매, 포인트 조회를 위한 서버 로그 |
| Media_LOGIN.csv | 로그인, 본인 인증, PIN 관리를 위한 서버 로그 |
| Media_MENU.csv | 초기 메뉴, 채널 카테고리 메뉴 제공을 위한 서버 로그 |
| Media_STREAM.csv | VOD 스트리밍을 위한 서버 로그 |

### 데이터 칼럼 설명

| columns | Explain |
| --- | --- |
| Timestamp | [YYYYMMDD_HHmm(a)-HHmm(b)] 형식을 가지며, 수집 범위는 YYYY년 MM월 DD일 HH시 mm분(a) 부터 HH시 mm분(b) |
| Request | 수집 범위 내 발생한 서비스 요청 수 |
| Success | 수집 범위 내 발생한 서비스 요청 성공 수 |
| Fail | 수집 범위내 발생한 서비스 요청 실패 수 |
| Session | 수집 시점의 미디어 스트리밍 세션 수 |
- Server-Prefix: 서비스를 제공하는 서버가 여러개일 경우, 각 서버의 번호가 컬럼명의 앞에 위치

## OTT 설명

- Over-The-Top의 약자
- KT에서 서비스하는 Seezn과 같이 인터넷을 통해 제공되는 각종 미디어 콘텐츠를 의미

## 제출 형식

- 정상(0)과 이상(1)을 판단하기 위한 모델을 학습
- 모델을 활용해, 2018년 전체에 대한 예측을 수행
- 제출 파일은 2018년 1월 1일 00시 00분-05분 부터 2018년 12월 31일 23시 55분-00분 구간의 이상 이벤트를 예측한 .csv 형식으로 저장
- 예측 데이터프레임의 크기는 [105120 * 1]

# EDA

## 라이브러리 및 데이터 로딩

데이터 전처리를 위해 numpy와 pandas 패키지 사용

```python
import numpy as np
import pandas as pd
```

그래프 시각화를 위해 matplotlib 패키지 사용

```python
from matplotlib import pyplot as plt
```

feature들의 scaling을 위해 sklearn의 MinMaxscaler, Robustscaler 로드

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler
```

데이터셋이 서버 카테고리별로 나뉘어져 있어 공통 path를 지정 후 load

```python
path = "/content/gdrive/MyDrive/KT/media/Media/"
 
data1 = pd.read_csv(path + "Media_INFO.csv")
data2 = pd.read_csv(path + "Media_LOGIN.csv")
data3 = pd.read_csv(path + "Media_MENU.csv")
data4 = pd.read_csv(path + "Media_STREAM.csv")
```

## 데이터 전처리

### 통합 데이터 생성
데이터가 4개의 csv파일로 각각 저장되어 있으므로 merge 함수를 사용하여 Timestamp를 기준으로 inner join을 진행하여 하나의 통합된 데이터를 생성

```python
all_data = pd.merge(data1, data2, on="Timestamp", how="inner")
all_data = pd.merge(all_data, data3, on="Timestamp", how="inner")
all_data = pd.merge(all_data, data4, on="Timestamp", how="inner")
```
### 필요없는 열 제거
feature별로 scaling을 해야 하는데, Timestamp는 scaling할 feature에 해당되지 않으므로 제거함. 또한, 이미 시간 순서대로 데이터가 나열되어 있으므로Timestamp열을 제거해도 무방

```python
all_data.drop(["Timestamp"], inplace=True, axis=1)
```
### 결측치 보간

![슬라이드5](https://user-images.githubusercontent.com/89781598/193593096-33c94ecb-be43-4e58-bad9-5ef646980500.JPG)


결측값이 많지 않으므로, 평균으로 결측치를 보간

```python
all_data = all_data.fillna(all_data.mean())
```

### Train Test Split

데이터를 카피하여 train 데이터와 test 데이터를 지정함. 이 때, train 데이터는 전체 데이터를 학습시킴. 그 이유는 일단 1차적으로 LSTM 오토인코더를 학습시킨 후, 임계값을 지정하여 1차적으로 이상치를 제거한 후, 이상치가 제거된 데이터를 다시 2차적으로 LSTM 오토인코더를 학습시켜서 최종모델을 완성하기 위함임.

```python
train = all_data.copy()
train.shape
```

test 데이터는 105120열부터 끝 열까지 이므로, 이를 따로 분리하여 test 데이터로 지정함.

```python
 
test = all_data[105120:]
test.reset_index(drop=True, inplace=True)
```

### 3차원 시퀀스 변환과 Partial Robust Scaling

![슬라이드6](https://user-images.githubusercontent.com/89781598/193593356-b816833c-f169-437a-9adf-772c999e32ef.JPG)


시퀀스 데이터로 저장하여 3차원 데이터로 데이터를 reshape함. 그 이유는 LSTM 오토인코더의 경우 시퀀스 데이터로 저장해야 시계열 데이터임을 인지할 수 있기 때문임. 이 때, timestep을 5로 지정함.

```python
# 3 차원 시퀀스 데이터로의 변경
 
def to_seq(df, step):
    output = []
    for i in range(len(df) - step - 1):
        temp = []
        for j in range(1, step + 1):
            temp.append(df.loc[[(i + j + 1)], :])
        output.append(temp)
    return np.squeeze(np.array(output))

# timestep을 5로 지정
step = abs(5)
trans_train = to_seq(train, step)
trans_train.shape
train_seq = trans_train
```

이 때, train 데이터를 RobustScaling을 하기 위해서 다시 2차원으로 축소시킨 후, 로버스트 스케일링을 진행함. 로버스트 스케일링을 진행한 이유는, 최대한 이상치를 반영하지 않고 스케일링을 진행하여, 1차 모델에서 LSTM 오토인코더 모델로 최대한 정상데이터만 추출해내기 위함임.

```python
# 2차원으로 차원을 축소시키는 함수 생성
def dimension_down(df):
    dimension_df = np.empty((df.shape[0], df.shape[2]))
    for i in range(df.shape[0]):
        dimension_df[i] = df[i, (df.shape[1] - 1), :]
 
    return dimension_df
 
# 로버스트 스케일링을 위한 함수 생성
def scaling(df, scaler):
    for i in range(df.shape[0]):
        df[i, :, :] = scaler.transform(df[i, :, :])
 
    return df
 
# 로버스트 스케일링 진행
scaler = RobustScaler().fit(dimension_down(train_seq))
train_scaled = scaling(trans_train, scaler)
print(train_scaled.shape)
```

# 모델링

![슬라이드8](https://user-images.githubusercontent.com/89781598/193593455-12b9ebe8-ef9e-44f9-93e9-23ce9f1d0930.JPG)
![슬라이드9](https://user-images.githubusercontent.com/89781598/193593459-e3d314cf-f5e2-414f-a8fd-00f65cbc0fb0.JPG)
![슬라이드10](https://user-images.githubusercontent.com/89781598/193593461-eda1a743-9415-4092-a6b6-1cb719267a7b.JPG)
![슬라이드11](https://user-images.githubusercontent.com/89781598/193593464-6df5ed02-3a2d-42f6-af9c-e537b4e60a46.JPG)
![슬라이드12](https://user-images.githubusercontent.com/89781598/193593470-6f790cfd-0a0d-40cf-8c5c-dbb0577378e0.JPG)
![슬라이드13](https://user-images.githubusercontent.com/89781598/193593474-3f236d3c-2a11-45fc-8325-af7b71abe02e.JPG)
![슬라이드14](https://user-images.githubusercontent.com/89781598/193593477-0260aac0-4234-446d-9958-d457bb4eef9b.JPG)
![슬라이드15](https://user-images.githubusercontent.com/89781598/193593481-1fb8dcd5-795c-435f-b46d-623a17a8ff31.JPG)

## 1차 모델 구성 및 학습

### 변수 선택 및 모델 구축

우선 1차적인 모델을 구성하여 전체 데이터 셋에서 이상치를 제거하고자 하였으며, 모델은 아래와 같이 구성하였음.

![Untitled](https://user-images.githubusercontent.com/89781598/193592300-96c272c1-4439-4f03-9ec4-f1c65f6ba16b.png)

domain 지식이 부족하다고 생각하여 feature를 섣불리 더하고 빼기보다는 시계열 데이터를 sequence to sequence로 반환하는 LSTM AE model이 적합하다고 판단하였음. 또한, LSTM AE의 경우 autoencoder와 유사하게 encoder의 layer에 data를 차원을 축소시킨 후, decoder의 복원을 하게 되는데 이 때, LSTM layer를 사용해 기울기 소실에 어느정도 면역을 가지고 있는것이 차이점임. 마지막으로, 일반적으로 정상 data만을 학습시키지만, 우리의 경우 정상, 비정상 data의 분류가 불가능하기에 세미나 때 들은 kt 측의 설명을 기반으로 오류가 상당히 적다고 판단함. 때문에 1차 모델에 data 전체를 학습하고 reconstruction하여도 오류 데이터는 상당히 적기 때문에 학습 횟수가 상대적으로 적어 rmse도 어느정도 클 것으로 기대.

아래의 코드로 모델을 구성하였음.

```python
def lstm_autoencoder():
    feature_number = train_scaled.shape[2]
    lstm_ae = models.Sequential()
 
    # 인코더
    lstm_ae.add(layers.LSTM(128, activation="relu", input_shape=(step, feature_number), return_sequences=True))
    lstm_ae.add(layers.LSTM(64, activation="relu", return_sequences=False))
    lstm_ae.add(layers.RepeatVector(step))
 
    # 디코더
    lstm_ae.add(layers.LSTM(64, activation="relu", return_sequences=True))
    lstm_ae.add(layers.LSTM(128, activation="relu", return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(feature_number)))
 
    print(lstm_ae.summary())
    return lstm_ae
 
lstm_ae0 = lstm_autoencoder()
```

### 모델 학습 및 검증

epoch는 20, batch_size는 128, learning_rate는 0.001로 진행하였으며, validation data의 비율은 0.2로 설정함.

```python
# 파라미터 설정
 
epochs = 20
batch_size = 128
learning_rate = 0.001
validation = 0.2
 
# compile
 
lstm_ae0.compile(loss="mse", optimizer=optimizers.Adam(learning_rate))
 
# 모델 학습
 
history = lstm_ae0.fit(
    train_scaled,
    train_scaled,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
    ],
)
 
lstm_ae0.save('lstm_ae0.h5') # 모델 저장
```

loss값의 간단한 그래프와 loss 및 val_loss를 보고 학습이 잘 되었는지 확인함.

```python
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Valid Loss")
plt.legend()
plt.show()
```

### 이상치 제거

우리는 모델을 재학습시킬 데이터가 필요하므로, 최대한 정상인 데이터를 추출해야함. 그러므로, 우리가 LSTM 오토인코더로 추출한 값과 원래의 input값의 차이를 제곱한 후 평균을 낸 값을 LOSS라고 판단. 이 때 임계값은 0.2로 선정하여 LOSS가 0.2를 넘으면 오류라고 판단하고, 넘지 않으면 정상이라고 판단. 최대한 오류를 많이 잡아내고, 정상데이터만 남도록 함. 마지막으로, 오토인코더로 이상치라고 판단한 관측값들을 모두 제외한 값들을 r_train에 저장시킴.

```python
# train의 output값을 추출
train_predictions = lstm_ae0.predict(train_scaled)
# 임계값을 0.2로 설정
threshold = 0.2
# 전체 데이터에서 이상치를 추려내여 r_train에 저장
train_error = []
for i in range(len(train_scaled)):
    if np.sqrt(np.mean(np.power(train_predictions.reshape(train_predictions.shape[0], train_predictions.shape[1] * train_predictions.shape[2])[i] - train_scaled.reshape(train_predictions.shape[0], train_predictions.shape[1] * train_predictions.shape[2])[i], 2))) > threshold:
        train_error.append(1)
    else:
        train_error.append(0)
train_error = pd.DataFrame(train_error, columns=['Prediction'])
train_error.value_counts()
err_index = list(train_error[train_error['Prediction'] == 1].index)
r_train = train.drop(err_index)
r_train.reset_index(drop=True, inplace=True)
```

## 2차 모델 구성과 학습

### 2차 모델 구성과 학습을 위한 전처리

105120행부터 시작해서 끝 열까지를 test 데이터로 지정한 후, 전체 데이터에서 1차 모델에서 이상치를 제거한 데이터인 r_train 데이터와 test 데이터를 timestep이 5인 3차원 데이터로 변환시킴. 이후, 1차 모델 때와는 달리, MinMaxScaler를 사용하여 스케일링을 진행함.

```python
# 1차 모델에서 이상치를 제거한 r_train 데이터와 test 데이터를 timestep이 5인 3차원 데이터로 변환
trans_train = to_seq(r_train, step)
trans_test = to_seq(test, step)
print(trans_test.shape)
train_seq = trans_train
# MinMaxScaling 진행
scaler = MinMaxScaler().fit(dimension_down(train_seq))
train_scaled = scaling(trans_train, scaler)
test_scaled = scaling(trans_test, scaler)
print(train_scaled.shape)
print(test_scaled.shape)
```

### 변수 선택 및 모델 구축

우리는 전체데이터에서 1차 모델을 통해 이상치를 제거한 데이터를 가지고 2차 모델을 학습시킴. 이 때, 2차 모델의 경우에도 LSTM 오토인코더를 사용하였음. 이 후, 2차 모델에는 정상 추측 dataset만으로 학습시켜 최종적으로 test data에서 anormaly를 predict 함. LSTM AE model은 2차원 dataframe 형태가 아닌 3차원의 시간축을 가진 형태로 data를 넣어줘야 하기에 변환과 scaling하고, timestep은 5로 설정함.

이 때, 모델의 구성은 아래와 같음.

![Untitled 1](https://user-images.githubusercontent.com/89781598/193592348-ac6dfeb5-c2fa-4562-adeb-20011098ead6.png)

모델을 구성할 때 사용한 코드는 아래와 같음.

```python
def lstm_autoencoder2():
    feature_number = train_scaled.shape[2]
    lstm_ae = models.Sequential()
 
    # 인코딩
    lstm_ae.add(layers.LSTM(64, activation="relu", input_shape=(step, feature_number), return_sequences=True))
    lstm_ae.add(layers.LSTM(32, activation="relu", return_sequences=False))
    lstm_ae.add(layers.RepeatVector(step))
 
    # 디코딩
    lstm_ae.add(layers.LSTM(32, activation="relu", return_sequences=True))
    lstm_ae.add(layers.LSTM(64, activation="relu", return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(feature_number)))
 
    print(lstm_ae.summary())
    return lstm_ae
 
# 모델 지정
lstm_ae1 = lstm_autoencoder2()
```

### 모델 학습과 검증

epoch는 10, batch_size는 128, learning_rate는 0.001로 진행하고, validation data의 비율은 0.2로 설정함.

```python
# 파라미터 설정
 
epochs = 10
batch_size = 128
learning_rate = 0.001
validation = 0.2
 
# compile
 
lstm_ae1.compile(loss="mse", optimizer=optimizers.Adam(learning_rate))
 
# 모델 학습
 
history = lstm_ae1.fit(
    train_scaled,
    train_scaled,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
    ],
)
 
# 모델저장
lstm_ae1.save('lstm_ae1.h5')
```

## 2차 모델을 이용한 이상 탐지

이제 2차모델에서 Test 데이터에 존재하는 이상치를 찾아냄. 임계값은 0.1로 설정. 그 이유는 1차모델에서는 전체 데이터에 대한 이상치를 찾아냈지만, 2차 모델에서는 전체 데이터의 크기의 1/2인 test 데이터에 대한 이상치를 찾아내려고 하기 때문임. 이 때, 1차 모델과 같은 방법으로 이상치를 찾아냄.

```python
# 예측값 찾아내기
test_predictions = lstm_ae1.predict(test_scaled)
 
# 임계 값 0.1로 설정
threshold = 0.1
 
# 예측값 저장
predict = []
for i in range(len(test_scaled)):
    if np.sqrt(np.mean(np.power(test_predictions.reshape(test_predictions.shape[0], test_predictions.shape[1] * test_predictions.shape[2])[i] - test_scaled.reshape(test_predictions.shape[0], test_predictions.shape[1] * test_predictions.shape[2])[i], 2))) > threshold:
        predict.append(1)
    else:
        predict.append(0)
 
for _ in range(step+1):
    predict.append(predict[-1])
 
predict = pd.DataFrame(predict, columns=['Prediction'])
predict.value_counts()
predict[predict['Prediction']==1]
 
# 정답 제출
predict.to_csv('predict.csv', mode='w')
```

## 결과 및 결언


![슬라이드17](https://user-images.githubusercontent.com/89781598/193593629-5c0cbb1b-98d7-4648-9b90-a9c572c0590e.JPG)

- 2개의 오토인코더를 학습하여 첫번째 오토인코더로는 전체 데이터에서 이상치를 제거하여 정상 데이터만을 추출하고, 두번째 오토인코더로는 정상인 데이터로만 학습시켜 이상치를 더욱 잘 탐지하도록 함.
- LSTM layer의 장점을 이용해 sequence를 더욱 길게 묶을 수 있게 step을 늘리는 방법을 사용하여 성능을 향상시키고 싶었으나, 컴퓨터의 성능 문제로 인해 진행할 수 없었음. 향후에 더 좋은 환경에서는 sequence를 더욱 길게 묶어서 이상치를 더 잘 탐지할 수 있을 것이라 기대.
- model에 drop layer를 추가하거나, regularize 하여 모델의 과적합을 방지해주는 방법도 시행해 볼 필요가 있음.
