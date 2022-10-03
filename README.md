# 제 1회 네트워크 지능화를 위한 인공지능 해커톤

## 개요
* 본 대회는 KT가 주관하고 ETRI(한국전자통신연구원)이 주최한 대회입니다.
* 총 2개의 과제를 수행해야 했습니다.
  * IP 네트워크 : 실시간 IP 할당 개수 추이 기반 이상 발생 시점 탐지
  * 미디어 서비스 : 실시간 OTT 서비스 이용자 수 추이 기반 이상 발생 시점 탐지
* 팀명은 부산대학교 DA팀이었습니다.

## 배경
* 본 대회는 예선과 본선으로 이루어져있었으며, 예선은 정량평가로 진행되었습니다.
* 예선에서의 정량평가와 본선에서의 발표를 통한 정성평가로 최종 순위를 확정지었습니다.

## 개발환경
<p align="center">
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <br>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>&nbsp
  <br>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/></a>&nbsp
</p>

## 프로젝트 결과
```
우수상 수상
```

![image](https://user-images.githubusercontent.com/89781598/193581270-79af4eb2-8ba2-49ed-a84c-2ea57ebd22a4.png)
![image](https://user-images.githubusercontent.com/89781598/193581326-8035eb0e-8461-4126-ab44-c6724958c15d.png)

```
시상식
```

![image](https://user-images.githubusercontent.com/89781598/193582696-fd846446-1144-4ea0-b2b6-c577ae0324e8.png)
![image](https://user-images.githubusercontent.com/89781598/193582721-488b4645-60b9-4da1-bb86-aebb865b6e6d.png)
![image](https://user-images.githubusercontent.com/89781598/193582745-a979e1fc-5999-4c73-849c-99ad4bc66d39.png)
![image](https://user-images.githubusercontent.com/89781598/193581406-82aeac14-9178-4657-b06f-8597c38e4efe.png)



## 파일 구조
```
📦LG.RADER
 ┣ 📂EDA
 ┃ ┣ 📜ANOVA.ipynb
 ┃ ┣ 📜Correlation and Distribution.ipynb
 ┃ ┗ 📜NULL.ipynb
 ┣ 📂MODEL
 ┃ ┣ 📜MultiOutput_Catboost_Bayesian.ipynb
 ┃ ┣ 📜MultiOutput_LGBM_Bayesian.ipynb
 ┃ ┗ 📜Submission.ipynb
 ┣ 📂Trial
   ┣ 📂AutoEncoder
   ┃ ┗ 📜AutoEncoder.ipynb
   ┣ 📂LSTM
   ┃ ┣ 📜LSTM_for_Submit.ipynb
   ┃ ┗ 📜LSTM_for_Validation.ipynb
   ┣ 📂SDCFEModel
   ┃ ┣ 📜SDCFEModel_for_Submit.ipynb
   ┃ ┗ 📜SDCFEModel_for_Validation.ipynb
   ┗ 📂UNET-Ensemble
   ┗ 📜UNET-Ensemble_for_validation.ipynb
```

## 파일 
- EDA : 통계적인 기법을 사용하여 데이터를 분석한 내용에 대해서 다룹니다.
    - ANOVA.ipynb : 57개의 features가 14개의 targets에 영향을 미치는지를 분석하기 위해서 ANOVA 검정을 시행하였다. 이 때, 수치형변수들을 모두 범주화하였다.
    - Correlation and Distribution.ipynb : features와 targets의 분포와 상관성을 조사하였다.
    - NULL.ipynb : 결측값을 확인하고 보간하였다.
    
- MODEL : 최종적으로 가장 성능이 우수한 모델에 대한 코드입니다.
    - MultiOutput_Catboost_Bayesian.ipynb : MultiOutput Catboost 모델에 대해서 베이지안 옵티마이저로 하이퍼 파라미터 튜닝을 진행하였으며, 결과가 잘 나온 2개의 모델을 선정하였다.
    - MultiOutput_LGBM_Bayesian.ipynb : MultiOutput LGBM 모델에 대해서 베이지안 옵티마이저로 하이퍼 파라미터 튜닝을 진행하였으며, 결과가 잘 나온 2개의 모델을 선정하였다.
    - Submission.ipynb : MultiOutput_Catboost_Bayesian.ipynb과 MultiOutput_LGBM_Bayesian.ipynb에서 선정한 4개의 모델을 앙상블하여 최종 결과에 반영하였다.이 때, 각각의 모델에서 제일 잘 나온 모델에 2/3의 가중치를, 그 다음으로 잘 나온 모델에 1/3가중치를 부여하였다.
    
- Trial : 문제를 해결하기 위해 여러 모델링을 시도한 코드입니다.
    - AutoEncoder : AutoEncoder를 이용하여 모델을 구성해보았습니다.
    - LSTM : LSTM layer를 이용하여 장기기억을 끌고가면서 모델을 학습시키면 성능이 더욱 좋을 것이라고 판단하여 이를 이용해 모델을 구성해보았습니다.
    - SDCFEModel
        - Seperate Dense Concat Fold Ensenble Model
        - 연관이 있는 열을 하나로 묶고, 연관성이 있는 그룹들 각각을 Dense layer로 차원을 축소하여 잠재벡터를 만듦.
        - 이 후, 해당 잠재벡터들을 모두 병합한 후, KFOLD 5로 5개의 모델을 각각 생성한 후, 모델들을 저장하고 Ensemble Method를 사용하여 5개의 모델을 모두 앙상블학습 시킴.
    - UNET-Ensemble
        - UNET의 구조를 착안하여 만든 모델로, UNET의 구조처럼 잠재벡터를 단계적으로 만들어나가면서, 단계별로 잠재벡터를 형성한다.
        - UNET의 최하단에 진입했을 때, 다시 상단으로 올라가면서 예측값을 생성한다.
        - 이 때, 상단으로 올라오는 단계별로 하단에서 만든 잠재벡터들을 아래에서 위로 끌어오면서, 모델이 하단으로 가면서 잃어버린 정보량을 최대한 보전하려고 노력한다.
        - 이 후, 최상단에서는 최하단으로 진입하는 단계에서 단계적으로 생성한 잠재벡터들을 모두 고려한 결과값을 도출한다.

## 참고사항
- 데이터의 경우 대회 종료 후 파기가 원칙이므로 제공되지 않습니다.

## 문의사항
* email : ajc227ung@gmail.com

