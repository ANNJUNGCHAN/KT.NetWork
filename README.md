# 제 1회 네트워크 지능화를 위한 인공지능 해커톤

![슬라이드1](https://user-images.githubusercontent.com/89781598/193593805-2dbec4b1-1639-4a15-a929-a834782d087a.JPG)

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

## 파일 구조
```
📦Code
 ┣ 📂IP
 ┃ ┣ 📜IP_모델파일.pt
 ┃ ┗ 📜IP_코드.ipynb
 ┗ 📂OTT
 ┃ ┣ 📜Media_이상탐지모델.h5
 ┃ ┣ 📜Media_전처리모델.h5
 ┃ ┗ 📜Media_코드.ipynb
```

## 파일 
- IP : 실시간 IP 할당 개수 추이 기반 이상 발생 시점 탐지에 대한 코드와 모델을 저장해놓았습니다.
    - IP_모델파일.pt : pytorch로 개발한 실시간 IP 할당 개수 추이 기반 이상 발생 시점 탐지 모델을 pt파일로 저장하였습니다.
    - IP_코드.ipynb : pytorch로 모델을 개발한 코드입니다.
- OTT : 실시간 OTT 서비스 이용자 수 추이 기반 이상 발생 시점 탐지에 대한 코드와 모델을 저장해놓았습니다.
    - Media_이상탐지모델.h5 : tensorflow로 개발한 실시간 OTT 서비스 이용자 수 추이 기반 이상 발생 시점 탐지모델을 h5파일로 저장하였습니다.
    - Media_전처리모델.h5 : 1차적으로 정상인 값만 filtering 해주는 LSTM AutoEncoder 모델을 h5파일로 저장하였습니다.
    - Media_코드.ipynb : tensorflow로 모델을 개발한 코드입니다.

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


## 참고사항
- 데이터의 경우 대회 종료 후 파기가 원칙이므로 제공되지 않습니다.
- 코드에 대한 자세한 설명은 각각의 주제에 대한 파일의 readme를 참고해주세요!

## 문의사항
* email : ajc227ung@gmail.com

