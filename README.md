Generative ABA

# ============================================================
# ABA ACTIVITY
# GENERATIVE MODEL VS DISCRIMINATIVE MODEL
# SMS SPAM DETECTION USING MACHINE LEARNING
# ============================================================

import os
import re
import time
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. DOWNLOAD DATASET
# ============================================================

print("Downloading SMS Dataset...")

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

os.makedirs("sms_dataset", exist_ok=True)

zip_path = "sms_dataset/sms.zip"
data_path = "sms_dataset/SMSSpamCollection"

if not os.path.exists(data_path):

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        filename = next(
            f for f in z.namelist()
            if f.endswith("SMSSpamCollection")
        )

        with open(data_path, "wb") as file:
            file.write(z.read(filename))

print("Dataset downloaded successfully!")

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("\nDATASET INFORMATION")
print("-------------------")

print("Total Messages:", len(df))

print("\nDataset Preview:")
print(df.head())

print("\nClass Distribution:")
print(df["label"].value_counts())

# ============================================================
# 3. DATA CLEANING
# ============================================================

df.dropna(inplace=True)

df.drop_duplicates(
    subset=["message"],
    inplace=True
)

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\.\S+",
        " URL ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_message"] = df["message"].apply(clean_text)

# Convert labels into numerical values
df["target"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nDATA CLEANING COMPLETED")

# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X = df["clean_message"]
y = df["target"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", len(X_train_text))
print("TESTING DATA:", len(X_test_text))

# ============================================================
# 5. FEATURE EXTRACTION USING TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = tfidf.fit_transform(X_train_text)

X_test = tfidf.transform(X_test_text)

print("\nFEATURE EXTRACTION COMPLETED")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

# ============================================================
# 6. GENERATIVE MODEL
# MULTINOMIAL NAIVE BAYES
# ============================================================

generative_model = MultinomialNB()

start_time = time.perf_counter()

generative_model.fit(
    X_train,
    y_train
)

generative_training_time = (
    time.perf_counter() - start_time
)

start_time = time.perf_counter()

generative_prediction = generative_model.predict(
    X_test
)

generative_prediction_time = (
    time.perf_counter() - start_time
)

print("\nGENERATIVE MODEL TRAINED SUCCESSFULLY")

# ============================================================
# 7. DISCRIMINATIVE MODEL
# LOGISTIC REGRESSION
# ============================================================

discriminative_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

start_time = time.perf_counter()

discriminative_model.fit(
    X_train,
    y_train
)

discriminative_training_time = (
    time.perf_counter() - start_time
)

start_time = time.perf_counter()

discriminative_prediction = discriminative_model.predict(
    X_test
)

discriminative_prediction_time = (
    time.perf_counter() - start_time
)

print("\nDISCRIMINATIVE MODEL TRAINED SUCCESSFULLY")

# ============================================================
# 8. PERFORMANCE EVALUATION
# ============================================================

def evaluate_model(name, actual, predicted,
                   training_time, prediction_time):

    return {
        "Model": name,

        "Accuracy": accuracy_score(
            actual, predicted
        ),

        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),

        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),

        "F1-score": f1_score(
            actual, predicted, zero_division=0
        ),

        "Training Time": training_time,

        "Prediction Time": prediction_time
    }


generative_results = evaluate_model(
    "Generative - Naive Bayes",
    y_test,
    generative_prediction,
    generative_training_time,
    generative_prediction_time
)

discriminative_results = evaluate_model(
    "Discriminative - Logistic Regression",
    y_test,
    discriminative_prediction,
    discriminative_training_time,
    discriminative_prediction_time
)

# ============================================================
# 9. COMPARISON TABLE
# ============================================================

results = pd.DataFrame([
    generative_results,
    discriminative_results
])

results = results.set_index("Model")

print("\n")
print("=" * 75)
print("GENERATIVE VS DISCRIMINATIVE MODEL COMPARISON")
print("=" * 75)

print(results.round(4).to_string())

# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nGENERATIVE MODEL REPORT")

print(classification_report(
    y_test,
    generative_prediction,
    target_names=["Ham", "Spam"],
    zero_division=0
))

print("\nDISCRIMINATIVE MODEL REPORT")

print(classification_report(
    y_test,
    discriminative_prediction,
    target_names=["Ham", "Spam"],
    zero_division=0
))

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

models_predictions = [
    ("Generative Model", generative_prediction),
    ("Discriminative Model", discriminative_prediction)
]

for ax, (name, prediction) in zip(
    axes,
    models_predictions
):

    cm = confusion_matrix(
        y_test,
        prediction
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ============================================================
# 12. PERFORMANCE COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

results[metrics].plot(
    kind="bar",
    figsize=(10, 5),
    ylim=(0, 1.05)
)

plt.title(
    "Generative vs Discriminative Model Performance"
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")

plt.xticks(rotation=10)

plt.legend(loc="lower right")

plt.tight_layout()

plt.show()

# ============================================================
# 13. TRAINING TIME COMPARISON
# ============================================================

results[
    ["Training Time", "Prediction Time"]
].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Training Time and Prediction Time Comparison")

plt.ylabel("Time in Seconds")

plt.xticks(rotation=10)

plt.tight_layout()

plt.show()

# ============================================================
# 14. TEST YOUR OWN SMS MESSAGE
# ============================================================

def predict_sms(message):

    cleaned = clean_text(message)

    features = tfidf.transform([cleaned])

    print("\nSMS MESSAGE:")
    print(message)

    gen_prediction = generative_model.predict(
        features
    )[0]

    disc_prediction = discriminative_model.predict(
        features
    )[0]

    print("\nGENERATIVE MODEL:")

    print(
        "SPAM" if gen_prediction == 1
        else "HAM (NORMAL MESSAGE)"
    )

    print("\nDISCRIMINATIVE MODEL:")

    print(
        "SPAM" if disc_prediction == 1
        else "HAM (NORMAL MESSAGE)"
    )


predict_sms(
    "Congratulations! You won a free prize. Call now!"
)

# ============================================================
# 15. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("ABA ACTIVITY COMPLETED SUCCESSFULLY")
print("=" * 60)

best_model = results["F1-score"].idxmax()

print("Best Model Based on F1-score:", best_model)

print("\nBoth models have been trained and evaluated.")
print("Performance comparison is displayed above.")


Downloading SMS Dataset...
Dataset downloaded successfully!

DATASET INFORMATION
-------------------
Total Messages: 5572

Dataset Preview:
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...

Class Distribution:
label
ham     4825
spam     747
Name: count, dtype: int64

DATA CLEANING COMPLETED

TRAINING DATA: 4135
TESTING DATA: 1034

FEATURE EXTRACTION COMPLETED
Training Features: (4135, 10000)
Testing Features: (1034, 10000)

GENERATIVE MODEL TRAINED SUCCESSFULLY

DISCRIMINATIVE MODEL TRAINED SUCCESSFULLY


===========================================================================
GENERATIVE VS DISCRIMINATIVE MODEL COMPARISON
===========================================================================
                                      Accuracy  Precision  Recall  F1-score  Training Time  Prediction Time
Model                                                                                                      
Generative - Naive Bayes                0.9642     1.0000  0.7176    0.8356         0.0033           0.0006
Discriminative - Logistic Regression    0.9584     0.9783  0.6870    0.8072         0.2085           0.0005

GENERATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       1.00      0.72      0.84       131

    accuracy                           0.96      1034
   macro avg       0.98      0.86      0.91      1034
weighted avg       0.97      0.96      0.96      1034


DISCRIMINATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.98      0.69      0.81       131

    accuracy                           0.96      1034
   macro avg       0.97      0.84      0.89      1034
weighted avg       0.96      0.96      0.96      1034
Downloading SMS Dataset...
Dataset downloaded successfully!

DATASET INFORMATION
-------------------
Total Messages: 5572

Dataset Preview:
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...

Class Distribution:
label
ham     4825
spam     747
Name: count, dtype: int64

DATA CLEANING COMPLETED

TRAINING DATA: 4135
TESTING DATA: 1034

FEATURE EXTRACTION COMPLETED
Training Features: (4135, 10000)
Testing Features: (1034, 10000)

GENERATIVE MODEL TRAINED SUCCESSFULLY

DISCRIMINATIVE MODEL TRAINED SUCCESSFULLY


===========================================================================
GENERATIVE VS DISCRIMINATIVE MODEL COMPARISON
===========================================================================
                                      Accuracy  Precision  Recall  F1-score  Training Time  Prediction Time
Model                                                                                                      
Generative - Naive Bayes                0.9642     1.0000  0.7176    0.8356         0.0033           0.0006
Discriminative - Logistic Regression    0.9584     0.9783  0.6870    0.8072         0.2085           0.0005

GENERATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       1.00      0.72      0.84       131

    accuracy                           0.96      1034
   macro avg       0.98      0.86      0.91      1034
weighted avg       0.97      0.96      0.96      1034


DISCRIMINATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.98      0.69      0.81       131

    accuracy                           0.96      1034
   macro avg       0.97      0.84      0.89      1034
weighted avg       0.96      0.96      0.96      1034
Downloading SMS Dataset...
Dataset downloaded successfully!

DATASET INFORMATION
-------------------
Total Messages: 5572

Dataset Preview:
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...

Class Distribution:
label
ham     4825
spam     747
Name: count, dtype: int64

DATA CLEANING COMPLETED

TRAINING DATA: 4135
TESTING DATA: 1034

FEATURE EXTRACTION COMPLETED
Training Features: (4135, 10000)
Testing Features: (1034, 10000)

GENERATIVE MODEL TRAINED SUCCESSFULLY

DISCRIMINATIVE MODEL TRAINED SUCCESSFULLY


===========================================================================
GENERATIVE VS DISCRIMINATIVE MODEL COMPARISON
===========================================================================
                                      Accuracy  Precision  Recall  F1-score  Training Time  Prediction Time
Model                                                                                                      
Generative - Naive Bayes                0.9642     1.0000  0.7176    0.8356         0.0033           0.0006
Discriminative - Logistic Regression    0.9584     0.9783  0.6870    0.8072         0.2085           0.0005

GENERATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       1.00      0.72      0.84       131

    accuracy                           0.96      1034
   macro avg       0.98      0.86      0.91      1034
weighted avg       0.97      0.96      0.96      1034


DISCRIMINATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.98      0.69      0.81       131

    accuracy                           0.96      1034
   macro avg       0.97      0.84      0.89      1034
weighted avg       0.96      0.96      0.96      1034





SMS MESSAGE:
Congratulations! You won a free prize. Call now!

GENERATIVE MODEL:
SPAM

DISCRIMINATIVE MODEL:
SPAM


============================================================
ABA ACTIVITY COMPLETED SUCCESSFULLY
============================================================
Best Model Based on F1-score: Generative - Naive Bayes

Both models have been trained and evaluated.
Performance comparison is displayed above.Downloading SMS Dataset...
Dataset downloaded successfully!

DATASET INFORMATION
-------------------
Total Messages: 5572

Dataset Preview:
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...

Class Distribution:
label
ham     4825
spam     747
Name: count, dtype: int64

DATA CLEANING COMPLETED

TRAINING DATA: 4135
TESTING DATA: 1034

FEATURE EXTRACTION COMPLETED
Training Features: (4135, 10000)
Testing Features: (1034, 10000)

GENERATIVE MODEL TRAINED SUCCESSFULLY

DISCRIMINATIVE MODEL TRAINED SUCCESSFULLY


===========================================================================
GENERATIVE VS DISCRIMINATIVE MODEL COMPARISON
===========================================================================
                                      Accuracy  Precision  Recall  F1-score  Training Time  Prediction Time
Model                                                                                                      
Generative - Naive Bayes                0.9642     1.0000  0.7176    0.8356         0.0033           0.0006
Discriminative - Logistic Regression    0.9584     0.9783  0.6870    0.8072         0.2085           0.0005

GENERATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       1.00      0.72      0.84       131

    accuracy                           0.96      1034
   macro avg       0.98      0.86      0.91      1034
weighted avg       0.97      0.96      0.96      1034


DISCRIMINATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.98      0.69      0.81       131

    accuracy                           0.96      1034
   macro avg       0.97      0.84      0.89      1034
weighted avg       0.96      0.96      0.96      1034





SMS MESSAGE:
Congratulations! You won a free prize. Call now!

GENERATIVE MODEL:
SPAM

DISCRIMINATIVE MODEL:
SPAM


============================================================
ABA ACTIVITY COMPLETED SUCCESSFULLY
============================================================
Best Model Based on F1-score: Generative - Naive Bayes

Both models have been trained and evaluated.
Performance comparison is displayed above.
Downloading SMS Dataset...
Dataset downloaded successfully!

DATASET INFORMATION
-------------------
Total Messages: 5572

Dataset Preview:
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...

Class Distribution:
label
ham     4825
spam     747
Name: count, dtype: int64

DATA CLEANING COMPLETED

TRAINING DATA: 4135
TESTING DATA: 1034

FEATURE EXTRACTION COMPLETED
Training Features: (4135, 10000)
Testing Features: (1034, 10000)

GENERATIVE MODEL TRAINED SUCCESSFULLY

DISCRIMINATIVE MODEL TRAINED SUCCESSFULLY


===========================================================================
GENERATIVE VS DISCRIMINATIVE MODEL COMPARISON
===========================================================================
                                      Accuracy  Precision  Recall  F1-score  Training Time  Prediction Time
Model                                                                                                      
Generative - Naive Bayes                0.9642     1.0000  0.7176    0.8356         0.0033           0.0006
Discriminative - Logistic Regression    0.9584     0.9783  0.6870    0.8072         0.2085           0.0005

GENERATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       1.00      0.72      0.84       131

    accuracy                           0.96      1034
   macro avg       0.98      0.86      0.91      1034
weighted avg       0.97      0.96      0.96      1034


DISCRIMINATIVE MODEL REPORT
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       903
        Spam       0.98      0.69      0.81       131

    accuracy                           0.96      1034
   macro avg       0.97      0.84      0.89      1034
weighted avg       0.96      0.96      0.96      1034

SMS MESSAGE:
Congratulations! You won a free prize. Call now!

GENERATIVE MODEL:
SPAM

DISCRIMINATIVE MODEL:
SPAM


============================================================
ABA ACTIVITY COMPLETED SUCCESSFULLY
============================================================
Best Model Based on F1-score: Generative - Naive Bayes

Both models have been trained and evaluated.
Performance comparison is displayed above.



SMS MESSAGE:
Congratulations! You won a free prize. Call now!

GENERATIVE MODEL:
SPAM

DISCRIMINATIVE MODEL:
SPAM


============================================================
ABA ACTIVITY COMPLETED SUCCESSFULLY
============================================================
Best Model Based on F1-score: Generative - Naive Bayes

Both models have been trained and evaluated.
Performance comparison is displayed above.




SMS MESSAGE:
Congratulations! You won a free prize. Call now!

GENERATIVE MODEL:
SPAM

DISCRIMINATIVE MODEL:
SPAM


============================================================
ABA ACTIVITY COMPLETED SUCCESSFULLY
============================================================
Best Model Based on F1-score: Generative - Naive Bayes

Both models have been trained and evaluated.
Performance comparison is displayed above.
