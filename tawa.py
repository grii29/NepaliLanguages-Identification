
import os
from sklearn.metrics import accuracy_score, classification_report
import ctypes
libTawa = ctypes.CDLL("/Users/Dell/Downloads/Tawa-main/Python/pyTawa.so")
from pyTawa.TLM import *
from pyTawa.TXT import *
from pyTawa.TAR import *

from tawa import load_model, codelength_string

tawa_models = {
    "Nepali": load_model("models/nepali.model"),
    "Bhojpuri": load_model("models/bhojpuri.model"),
    "Maithili": load_model("models/maithili.model"),
    "Newari": load_model("models/newari.model"),
    "Tamang": load_model("models/tamang.model"),
    "Tharu": load_model("models/tharu.model")
}

#Test data
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
    file_path = f"data/{filename}"
    if not os.path.exists(file_path):
        continue

#Tawa prediction function
def tawa_predict(texts):
    predictions = []
    for text in texts:
        codelengths = {lang: codelength_string(model, text) for lang, model in tawa_models.items()}
        pred = min(codelengths, key=codelengths.get)
        predictions.append(pred)
    return predictions

tawa_preds = tawa_predict(texts)

#Evaluate
acc = accuracy_score(labels, tawa_preds)
print(f"Tawa Accuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(labels, tawa_preds, digits=4))