from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
class PumpInput(BaseModel):
    water_level: float
    Temperature: float
    Humidity: float
    N: float
    P: float
    K: float
# Load model
model = joblib.load("xgb_pump_model.pkl")

@app.post("/predict")
def predict(
    input_data: PumpInput
):
    # Prepare input (order must match training data)
    input_data = np.array([[input_data.Temperature, input_data.Humidity, input_data.water_level, input_data.N, input_data.P, input_data.K]])
    
    # Predict
    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob > 0.5)
    
    return {
        "probability": float(prob),
        "prediction": prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)