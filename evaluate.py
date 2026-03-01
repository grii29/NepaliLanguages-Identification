import re
from collections import defaultdict

with open("results.txt", "r") as f:
    lines = f.readlines()

true_labels = []
predicted_labels = []

for line in lines:
    if "The lowest codelength" in line:
        # Extract true label
        true_match = re.search(r"label (\w+)", line)
        pred_match = re.search(r"tag (\w+)", line)

        if true_match and pred_match:
            true_labels.append(true_match.group(1))
            predicted_labels.append(pred_match.group(1))

classes = sorted(set(true_labels))

conf_matrix = {c: {c2: 0 for c2 in classes} for c in classes}

for t, p in zip(true_labels, predicted_labels):
    conf_matrix[t][p] += 1

print("\nConfusion Matrix:")
print("True\\Pred", *classes)
for c in classes:
    print(c, *[conf_matrix[c][c2] for c2 in classes])

print("\nMetrics per class:")
precisions = []
recalls = []
f1s = []

for c in classes:
    TP = conf_matrix[c][c]
    FP = sum(conf_matrix[other][c] for other in classes if other != c)
    FN = sum(conf_matrix[c][other] for other in classes if other != c)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(f"{c}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

macro_precision = sum(precisions) / len(classes)
macro_recall = sum(recalls) / len(classes)
macro_f1 = sum(f1s) / len(classes)

total_TP = sum(conf_matrix[c][c] for c in classes)
total = len(true_labels)
micro_precision = total_TP / total
micro_recall = total_TP / total
micro_f1 = total_TP / total

print("\nMacro Averages:")
print(f"Precision={macro_precision:.3f}, Recall={macro_recall:.3f}, F1={macro_f1:.3f}")

print("\nMicro Averages:")
print(f"Precision={micro_precision:.3f}, Recall={micro_recall:.3f}, F1={micro_f1:.3f}")

import json

tawa_metrics = {
    "accuracy": macro_precision,  
    "precision": macro_precision,
    "recall": macro_recall,
    "f1_score": macro_f1
}

import os
os.makedirs("models", exist_ok=True)

with open("models/tawa_metrics.json", "w") as f:
    json.dump(tawa_metrics, f)
