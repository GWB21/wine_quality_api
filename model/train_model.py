import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# 모델 저장 폴더 생성
os.makedirs('model', exist_ok=True)

# 데이터 로드
data = pd.read_csv('winequality-red.csv', sep=';')

# 특성과 타겟 분리
X = data.drop('quality', axis=1)
y = data['quality']

# 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 검증
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f'MSE: {mse:.4f}')

# 모델 저장
joblib.dump(model, 'model/wine_model.pkl')
print('모델이 model/wine_model.pkl에 저장되었습니다.')