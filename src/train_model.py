import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
df = pd.read_csv("data/processed/labels.csv")

# 2. feature / label 분리
X = df[["p_gain_mult"]]
y = df["stable"]

# 3. train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. 예측
y_pred = model.predict(X_test)

# 6. 평가
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# 7. 계수 확인 (중요!)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)