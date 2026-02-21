from flask import Flask, render_template, request
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

app = Flask(__name__)

#Load vectorizer and models
vectorizer = joblib.load("models/vectorizer.joblib")
model_names = ["MultinomialNB", "kNN", "DecisionTree", "RandomForest", "SVM"]
models = {name: joblib.load(f"models/{name}.joblib") for name in model_names}
accuracy_dict = joblib.load("models/accuracy_dict.joblib")


#Convert dict keys/values to Python lists for JS
labels = list(accuracy_dict.keys())
values = list(accuracy_dict.values())

#FLASK ROUTES

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_input = request.form.get("text_input")
        model_choice = request.form.get("model_choice")
        model = models[model_choice]
        X_input_vec = vectorizer.transform([text_input])
        prediction = model.predict(X_input_vec)[0]
        return render_template(
            "result.html",
            text=text_input,
            model=model_choice,
            prediction=prediction,
            labels = labels,
            values=values
            )

    return render_template("index.html", models=models.keys())

if __name__ == "__main__":
    app.run(debug=True)
