import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text.lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()

# Load and clean data
df = pd.read_csv("train.csv")
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_vec, y)
lr_acc = accuracy_score(y, lr_model.predict(X_vec))
print(f"LogisticRegression Accuracy: {lr_acc:.4f}")

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_vec, y)
dt_acc = accuracy_score(y, dt_model.predict(X_vec))
print(f"DecisionTree Accuracy: {dt_acc:.4f}")

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_vec, y)
rf_acc = accuracy_score(y, rf_model.predict(X_vec))
print(f"RandomForest Accuracy: {rf_acc:.4f}")
