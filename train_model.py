import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text.lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()

# Function to train, evaluate, and visualize a model
def train_and_evaluate(model_name, model, X_vec, y):
    model.fit(X_vec, y)
    y_pred = model.predict(X_vec)
    acc = accuracy_score(y, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    return model

# Load and clean data
df = pd.read_csv("train.csv")
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_acc = 0

# Loop through models, pick best
for name, mdl in models.items():
    trained = train_and_evaluate(name, mdl, X_vec, y)
    acc = accuracy_score(y, trained.predict(X_vec))
    if acc > best_acc:
        best_acc = acc
        best_model = trained

# Save best model and vectorizer
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print(f"Saved best model with accuracy: {best_acc:.4f}")
