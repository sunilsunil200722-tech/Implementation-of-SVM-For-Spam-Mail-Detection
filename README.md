# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Input: Load and read the spam dataset containing messages and their labels.
2.Processing: Clean the text, remove stopwords, and convert it into TF-IDF feature vectors.
3.Decision: Train the SVM classifier and use it to predict whether a message is spam or not.
4.Output: Display results such as accuracy, confusion matrix, and message prediction. 

## Program:
```import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Features and target
X = data['message']
y = data['label']

# TF-IDF conversion
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sunil Kumar R
RegisterNumber:  212225040440
*/


## Output:
<img width="816" height="624" alt="Screenshot 2026-03-17 231653" src="https://github.com/user-attachments/assets/7451192f-1d95-4470-8ba9-6dc16c88ee22" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
