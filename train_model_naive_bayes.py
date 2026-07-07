import pandas as pd
from app.preprocessing import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
import joblib

df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None,names=["label", "message"])

# apply preprocessing to all messages
df["message"] = df["message"].apply(preprocess)

# Split data into training and testing
X = df["message"]
y= df["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# vectorization
vectorizer = CountVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

# train model
model = MultinomialNB()
model.fit(X_train, y_train)

# save vectorizer
joblib.dump(vectorizer,"models/vectorizer.pkl")

#save the model
joblib.dump(model,"models/spam_classifier.pkl")

# testing
predictions = model.predict(X_test)

# print(model.classes_)

# # evaluation
print("Accuracy: ", accuracy_score(y_test, predictions),"\n")
print("Classification Report:\n", classification_report(y_test, predictions),"\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


