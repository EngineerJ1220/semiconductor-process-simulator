# Semiconductor Process Simulator

교육용 합성 데이터를 이용해 Sputter 공정의 이상 Lot과 원인을 진단하는 Streamlit 웹 애플리케이션입니다.

## v2.1.0 주요 기능

- Al, Cu, Ti, Ta 재료 선택
- 쉬움, 보통, 어려움, 전문가 난이도 선택
- Lot당 25장 Wafer 데이터 생성
- Power, Ar Flow, Pressure, 증착률, 두께, 균일도, 예상 면저항 생성
- 랜덤 이상 주입
- 신호 중첩, 센서 편향, 일시적 변동, Recipe 변경, 일부 Wafer 이상
- 전문가 난이도 복합 이상
- Random Forest 기반 AI 자동 진단
- AI 1순위·2순위 원인과 확률 표시
- 신뢰도 및 판단 유보
- 난이도별 정확도, Macro F1, 이상 Lot 정확 일치, ±1 Lot 탐지율
- 정상 Case 오탐률과 Top-2 원인 적중률
- 사람·통계 기반 탐지·AI 진단 비교
- Raw Data와 AI 학습용 특징 데이터 CSV 다운로드

## 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 주의

이 프로젝트는 실제 기업 Recipe나 생산 데이터를 사용하지 않은 교육용 합성 데이터 시뮬레이터입니다.
AI 성능은 이 시뮬레이터가 생성한 미사용 합성 Case에 대한 결과이며 실제 Fab 성능을 의미하지 않습니다.
