# SMS Spam Message Classifier

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **Multinomial Naive Bayes**. The project includes a FastAPI backend for predictions and a Streamlit web application for an interactive user interface.

---

## Live Demo

**Streamlit App:**  

## Application Preview



## Features

- Classifies SMS messages as **Spam** or **Ham**
- Text preprocessing pipeline
- CountVectorizer for text feature extraction
- Multinomial Naive Bayes classifier
- FastAPI REST API
- Streamlit web application
- Returns prediction with confidence score

---

## API

### POST `/predict`

Request

```json
{
    "message": "Free iPhone! Click here to claim your prize!"
}
```

Response

```json
{
    "prediction": "spam",
    "probability": 0.99
}
```
---

## Streamlit Application

The project also includes a local Streamlit application for interactive predictions.

Features:

- Enter any SMS message
- Predict Spam or Ham
- Display prediction confidence

## Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes
- FastAPI
- Streamlit

---

## Machine Learning / NLP Concepts
- Text preprocessing
- Natural Language Processing (NLP)
- CountVectorizer
- TF-IDF
- Multinomial Naive Bayes
- Logistic Regression
- Model comparison
- Precision, Recall and F1-score
- Confusion Matrix interpretation
- Saving and loading trained models
- Building prediction modules

---

## Running Locally

### Clone the repository

```bash
git clone https://github.com/jrupasinghe222-ux/spam-classifier.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit frontend

```bash
streamlit run streamlit_app.py
```
---

### Run the API locally
```bash
uvicorn app.main:app --reload
```