from fastapi import FastAPI
from app.predictor import predict_message
from app.schemas import MessageRequest

app = FastAPI(title="Spam/Ham Classifier API")

@app.get("/")
def home():
    return { "message": "Spam Classifier API is running!"}

@app.post("/predict")
def predict(request:MessageRequest):
    return predict_message(request.message)

