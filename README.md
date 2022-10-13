# 🌈 Ultraviolet-Prediction-Model
> 기상청에서 주최하는 『2022 날씨 빅데이터 콘테스트』 대회에 참가하여, 딥러닝 모델을 학습시킨 과정들을 기록한 저장소입니다.

# 🙋‍♀️ 참가
- 소프트웨어 및 시스템 보안 연구실 내 석,박사 과정 각 1명과, 학부생 2명(필자)으로 참가했다. <br>
# ⛅️ 소개 
- ## 주제 : 기상위성 자료를 활용한 여름철 자외선 산출 기술 개발
- #### 날씨 빅데이터와 딥러닝 모델을 활용하여 자외선 수치 예측해보기로 했다.

# 🏆 대회 진행 과정 
_대회를 진행하면서 전 과정들을 "[notion](https://www.notion.so/a315e332914c473eb5f45651a1346eea)"에 기록해두었습니다._
> #### 일정 
> - 참가신청 : 2022.05.30 (~ 06.08 연장)
> - 1차 미팅 : 2022.05.31(월) 13시
> - 2차 미팅 : 2022.06.20(월) 13시
> - 3차 미팅 : 2022.06.27(월) 13시
> - 4차 미팅 : 2022.07.04(월) 13시
> - 1차 대회 제출 : 2022.07.18(월) <br>
>   ( ~ 08.01(월) 17:00 연장 -> 08.03(수) 17:00 연장) <br><br>

## 임무분담
- 석사 조교님과 필자는 과제 1 모델(기상위성 자료를 활용한 여름철 자외선 산출 기술 개발)을 만들고, 
- 박사 조교님과 다른 학부생은 과제 2 모델(기상위성 자료를 활용한 지면/지상 온도 산출 기술 개발)을 만든다.
- 각자 모델을 만들어 가장 성능이 좋은 사람의 모델로 대회 제출을 한다.
## 1차 미팅 이후

- 날씨마루 가입 및 데이터 권한 요청.
- 과제 파악.
- 데이터 분석 요인 및 데이터셋 찾아보고 데이터 품질 검토.
- 어떤 딥러닝 모델을 사용하면 좋을지 결론 내리기.

## 2차 미팅 이후
- [데이터 분석하기](https://www.notion.so/e74446fafbb543d19990f525f25c27ae) 
- [데이터 처리하기](https://www.notion.so/0c8bb7ee42f64ec3aedad4a8d4271586)
- [딥러닝 모델에 대해 공부하기 - RNN 편](https://www.notion.so/RNN-72dd5fe8fad34c9b9babd2569a5a8f3b)
- [딥러닝 모델에 대해 공부하기 - LSTM 편](https://www.notion.so/LSTM-2cbc9d86874047e484bb4fa2eff25069)
- [학습 모델 관련 지식 쌓기](https://www.notion.so/bb46aea4fa9c43348a3c748767edb147)


## 3차 미팅 이후
- 발전 방향 토의 및 수행하기
- 데이터 간의 상관관계 파악을 위해 correlation 돌려보기 
- 지점별로 학습시키보기 
- [모델 개선하기](https://www.notion.so/7c280cb1cf914340998543759bb40691)
## 4차 미팅 이후
- [칼럼값 변경 후 위치별로 모델 예측률 평가하기](https://www.notion.so/65cc31eaea7c446ca0eabd579e65fae6)
- [모델 정확도 향상시켜보기](https://www.notion.so/154476febef74aa581810034b9fd3d6a)
- tensorflow -> [pytorch](https://www.notion.so/torch-d273c8b5fed245a99b8a78880158c7ce)로 바꾸어 모델 다시 작성

<br>


<details>
<summary> 참고한 자료 </summary>
<div markdown="1">

✅ [공모전 사이트](https://bd.kma.go.kr/contest/)

✅  과제 영상 
- [1-1](https://www.youtube.com/watch?v=R_fa3HxPYdw)
- [1-2](https://www.youtube.com/watch?v=_r2WjDsYoqM)

✅ 데이터 및 분석 사이트  

→ [날씨마루(메인)](https://bd.kma.go.kr/)

→ [기상자료개방포털](https://data.kma.go.kr/cmmn/main.do)

→ [공공데이터포털](http://data.go.kr/)

→ [기상 자료](https://nmsc.kma.go.kr/homepage/html/base/cmm/selectPage.do?page=static.edu.satelliteClsf)

→ JOISS 해양 데이터 포털

</div>
</details>

<br>

# 🚨 문제 발생과 해결 방안
(작성중)
# 💭 느낀점 및 개선하고 싶은 점
(작성중)


딥러닝 모델에 대해 

홈페이지에서 검증을 하는 데 홈페이지에서의 서버 오류로 인해 검증을 제대로 하지 못했다. 
대회 마감일이 연장 되는 등 여러 우여곡절 끝에 제출 할 수 있었다.

만든 모델의 정확도가 높지 못했다. 
