# Drone Stability Prediction (drone-stability-ml)

드론 시뮬레이터(gym-pybullet-drones)를 기반으로  
제어 파라미터(P gain)를 변화시키며 비행 안정성을 분석하고,  
그 결과를 머신러닝 모델로 예측하는 프로젝트입니다.

---

## 📌 Project Overview

- 시뮬레이션 기반 데이터 생성
- 안정성(Stable / Unstable) 라벨링
- 머신러닝 모델을 통한 안정성 예측

---

## ⚙️ Pipeline

Simulation → Log Data → Labeling → ML Model → Visualization

---

## 📂 Project Structure

~~~
src/                # simulation, labeling, training code
data/processed/     # labeled dataset
results/logs/       # simulation logs
results/plots/      # visualization results
configs/            # workload configuration
~~~

---

## 🚀 How to Run

### 1. 시뮬레이션 실행

~~~
python src/simulate.py --p_gain_mult 1.0 --duration 5 --seed 42 --output results/logs/run_test.csv
~~~

### 2. batch 실행

~~~
python src/run_batch.py
~~~

### 3. 라벨링

~~~
python src/label_stability.py
~~~

### 4. 모델 학습

~~~
python src/train_model.py
~~~

---

## 📊 Example Result

- p_gain 값에 따라 안정성(Stable / Unstable)이 달라지는 경향 확인  
- 결과 그래프: `results/plots/p_gain_vs_stability.png`

---

## 🧪 Current Status

- p_gain 기반 시뮬레이션 및 로그 생성 완료  
- 안정성 라벨링 완료  
- Logistic Regression baseline 모델 학습 완료  

---

## 🔗 Reference

- https://github.com/utiasDSL/gym-pybullet-drones
