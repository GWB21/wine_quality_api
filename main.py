from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Wine Quality Prediction API")

# 입력 데이터 모델 정의
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

# 모델 로드
try:
    model = joblib.load('model/wine_model.pkl')
except Exception as e:
    raise Exception(f"모델을 로드하는데 실패했습니다: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "와인 품질 예측 API에 오신 것을 환영합니다!"}

@app.post("/predict")
def predict_wine_quality(wine: WineFeatures):
    try:
        # 입력 데이터를 numpy 배열로 변환
        features = np.array([[
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.citric_acid,
            wine.residual_sugar,
            wine.chlorides,
            wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.ph,
            wine.sulphates,
            wine.alcohol
        ]])
        
        # 예측 수행
        prediction = model.predict(features)[0]
        
        return {
            "predicted_quality": round(prediction, 2),
            "message": "예측이 성공적으로 완료되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 