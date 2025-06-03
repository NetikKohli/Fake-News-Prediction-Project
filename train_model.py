import pandas as pd
import re
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load real and fake news datasets, and assign binary labels
df_real = pd.read_csv("real_news.csv")    # Real news articles
df_fake = pd.read_csv("fake_news.csv")    # Fake news articles

df_real["label"] = 0  # Label real news as class 0
df_fake["label"] = 1  # Label fake news as class 1

# Combine both datasets into a single DataFrame
df = pd.concat([df_real, df_fake], ignore_index=True)

# Remove exact duplicates based on raw text
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Basic text cleaning function
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Remove URLs
    text = re.sub(r'<.*?>', ' ', text)                  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z ]', ' ', text)             # Remove non-alphabetic characters
    words = text.split()
    return ' '.join(words)  # Join cleaned words back into a string

# Apply text cleaning to all news articles
df["text"] = df["text"].apply(clean_text)

# Remove duplicates again now that text has been cleaned
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Prepare features and labels
X = df["text"]
y = df["label"]

# Perform a stratified train-test split to maintain label balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Sanity check to ensure no text appears in both train and test sets
overlap = set(X_train).intersection(set(X_test))
print(f"Exact overlap between train and test sets: {len(overlap)} (should be 0)")

# Convert text into TF-IDF vectors for machine learning input
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define several candidate models for comparison
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

best_model = None
best_acc = 0.0

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)            # Train on training set
    y_pred = model.predict(X_test_vec)         # Predict on test set
    acc = accuracy_score(y_test, y_pred)       # Calculate accuracy
    print(f"{name} Accuracy: {acc:.4f}")

    # Plot and display confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Real", "Fake"], cmap=plt.cm.Blues
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.show()

    # Update best model if current model performs better
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\n Best Model: {type(best_model).__name__} with Accuracy {best_acc:.4f}")

# Persist the best model and vectorizer to disk for later use
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
