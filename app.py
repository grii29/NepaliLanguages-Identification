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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

app = Flask(__name__)

#Load vectorizer and models
vectorizer = joblib.load("models/vectorizer.joblib")
model_names = ["MultinomialNB", "kNN", "DecisionTree", "RandomForest", "SVM"]
models = {name: joblib.load(f"models/{name}.joblib") for name in model_names}
metrics_dict = joblib.load("models/metrics_dict.joblib")


labels = list(metrics_dict.keys())
accuracy_values = [metrics_dict[name]['accuracy'] for name in labels if 'accuracy' in metrics_dict[name]]
f1_values = [metrics_dict[name]['f1_score'] for name in labels]
precision_values = [metrics_dict[name]['precision'] for name in labels]
recall_values = [metrics_dict[name]['recall'] for name in labels]

#FLASK ROUTES
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_input = request.form.get("text_input")
        model_choice = request.form.get("model_choice")
        model = models[model_choice]
        X_input_vec = vectorizer.transform([text_input])
        prediction = model.predict(X_input_vec)[0]
        model_metrics = metrics_dict.get(model_choice, {})

        return render_template(
            "result.html",
            text=text_input,
            model=model_choice,
            prediction=prediction,
            accuracy=model_metrics.get("accuracy", 0),
            precision=model_metrics.get("precision", 0),
            recall=model_metrics.get("recall", 0),
            f1_score=model_metrics.get("f1_score", 0),
            labels=labels or [],
            accuracy_values=accuracy_values or [],
            f1_values=f1_values,
            precision_values=precision_values,
            recall_values=recall_values
            )

    return render_template("index.html", models=models.keys())

if __name__ == "__main__":
    app.run(debug=True)
