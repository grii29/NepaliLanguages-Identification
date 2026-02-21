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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    "kNN": KNeighborsClassifier(n_neighbors=5).fit(X_train_vec, y_train),
    "DecisionTree": DecisionTreeClassifier(random_state=42).fit(X_train_vec, y_train),
    "RandomForest": RandomForestClassifier(n_estimators=5, random_state=42).fit(X_train_vec, y_train),
    "SVM": LinearSVC(random_state=42, max_iter=5000).fit(X_train_vec, y_train)
}
#Evaluate metrics
metrics_dict = {}
for name, model in models.items():
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics_dict[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

#Print metrics
for name, metrics in metrics_dict.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

#Save models and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.joblib")
for name, model in models.items():
    joblib.dump(model, f"models/{name}.joblib")
joblib.dump(metrics_dict, "models/metrics_dict.joblib")

print("Training complete and models saved!")
