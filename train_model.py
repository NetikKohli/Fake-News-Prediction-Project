import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("train.csv")
X = df['text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Evaluate on training data (initial check)
preds = model.predict(X_vec)
print("Accuracy:", accuracy_score(y, preds))
