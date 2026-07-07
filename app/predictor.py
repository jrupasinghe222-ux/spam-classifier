import joblib
import os
from app.preprocessing import preprocess

# Load trained model and vectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTORIZER_PATH = os.path.join(BASE_DIR,"models","vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR,"models","spam_classifier.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


def predict_message(message:str):

    """
    Predict whether a message is spam or ham.

    Returns:
        dict containing:
        - prediction
        - the probability of message belonging to predicted class
    """

    # preprocess
    message = preprocess(message)

    # vectorize
    message_vector = vectorizer.transform([message])

    # predict
    prediction = model.predict(message_vector)[0]

    # get probabilities
    probabilities = model.predict_proba(message_vector)[0]

    if prediction == "spam":
        probability = probabilities[1]
    else:
        probability = probabilities[0]


    return {
        "prediction":prediction,
        "probability":round(probability,2)
    }
