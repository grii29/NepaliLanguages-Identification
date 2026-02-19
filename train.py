import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load data
label_map = {
    "nepali_cleaned.txt": "Nepali",
    "bhojpuri_cleaned.txt": "Bhojpuri",
    "maithili_cleaned.txt": "Maithili",
    "newari_cleaned.txt": "Newari",
    "tamang_cleaned.txt": "Tamang",
    "tharu_cleaned.txt": "Tharu"
}

texts, labels = [], []
for filename, label in label_map.items():
    if not os.path.exists(f"data/{filename}"):
        continue
    with open(f"data/{filename}", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(label)

#Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_df = pd.DataFrame({
    'text': X_train,
    'label': y_train
})

#Convert test set to a DataFrame
test_df = pd.DataFrame({
    'text': X_test,
    'label': y_test
})

#Save to files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

#TF-IDF vectorizer 
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,4), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train models 
models = {
    "MultinomialNB": MultinomialNB().fit(X_train_vec, y_train),
    "kNN": KNeighborsClassifier(n_neighbors=3).fit(X_train_vec, y_train),
    "DecisionTree": DecisionTreeClassifier(random_state=42).fit(X_train_vec, y_train),
    "RandomForest": RandomForestClassifier(n_estimators=5, random_state=42).fit(X_train_vec, y_train),
    "SVM": LinearSVC(random_state=42, max_iter=5000).fit(X_train_vec, y_train)
}

#Compute accuracies
accuracy_dict = {name: float(accuracy_score(y_test, model.predict(X_test_vec)))
                 for name, model in models.items()}

#Save models and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.joblib")
for name, model in models.items():
    joblib.dump(model, f"models/{name}.joblib")
joblib.dump(accuracy_dict, "models/accuracy_dict.joblib")

print("Training complete and models saved!")
