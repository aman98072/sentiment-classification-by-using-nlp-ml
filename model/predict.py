import joblib
from model.preprocessing import clean_text

model = joblib.load("artifacts/model.pkl")
vectorizer = joblib.load("artifacts/vectorizer.pkl")

def predict_sentiment(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    return {
        "label": pred,
        "confidence": float(prob)
    }