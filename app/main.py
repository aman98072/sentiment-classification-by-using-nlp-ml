from fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predict_sentiment

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: TextRequest):
    result = predict_sentiment(req.text)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)