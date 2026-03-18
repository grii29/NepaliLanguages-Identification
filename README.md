NepLID: Nepali Mother Language Identification NepLID is a tool to identify the mother language of Nepali text. 

It supports Devnagari and Romanized scripts and can classify text into multiple languages such as Nepali, Bhojpuri, Newari, Tamang, Maithili, Tharu, etc. 

Features:
Predicts language from Devnagari or Romanized Nepali text.
-Supports multiple classification algorithms (MultinomialNB, kNN, DecisionTree, RandomForest, SVM, TAWA)
-Maintains a history of predictions using SQLite database.
-Accepts inputs via CSV for testing.
Getting Started
Prerequisites
Python 3.10+
pip packages
Git LFS (for large model files)

Files
devnagarari_test.csv – sample input file in the devnaagri script for batch prediction.
roman_test.csv - sample input file in roman/latin script for batch prediction.
app.py – Flask application.
models/ – folder containing trained models and vectorizers.
neplid.db – SQLite database for storing prediction history.

Usage
1. Running the Flask app
python app.py

Open your browser and go to http://127.0.0.1:5000
Select Devnagari or Romanized input.
Choose a model and input text directly to predict.
Click Predict Language.

2. Using devnagari_test.csv & roman_test.csv for predictions
