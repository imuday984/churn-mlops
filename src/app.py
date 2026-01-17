from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# 1. Load the model on startup
model = joblib.load('model.pkl')

# 2. Define the Input Data Format (Schema)
# This ensures users send the right data types
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input JSON to DataFrame (User 1 row)
    input_data = pd.DataFrame([data.dict()])
    
    # Rename 'gender' to 'Gender' to match training data exact spelling
    input_data.rename(columns={'gender': 'Gender'}, inplace=True)
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of Churn (1)

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": float(probability)
    }

# Run with: uvicorn src.app:app --reload