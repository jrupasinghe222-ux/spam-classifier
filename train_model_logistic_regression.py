import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from app.preprocessing import preprocess
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline

df =  pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

# preprocessing
df["message"] = df["message"].apply(preprocess)

# train, test split
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


pipeline = Pipeline([
    ("vectorizer", CountVectorizer(stop_words="english")),
    ("classifier", LogisticRegression())
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

# evaluation
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print()

print("Classification Report:")
print(classification_report(y_test, predictions))
print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

