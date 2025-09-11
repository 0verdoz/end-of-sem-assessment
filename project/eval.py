import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
import os

from project.preprocess import stemming

# Create reports directory if it doesn't exist
os.makedirs('./reports', exist_ok=True)

# Load preprocessed test data and model
x_test = np.load('./data/x_test.npy', allow_pickle=True)
y_test = np.load('./data/y_test.npy', allow_pickle=True)
model = joblib.load('./models/logistic_regression_model.joblib')

# Predict
x_test_prediction = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(x_test_prediction, y_test)
# precision, recall, f1, _ = precision_recall_fscore_support(y_test, x_test_prediction, average='binary') #average='macro'
precision, recall, f1, _ = precision_recall_fscore_support(y_test, x_test_prediction, average='binary') 
cm = confusion_matrix(y_test, x_test_prediction).tolist()  # Convert to list for JSON serialization

# Prepare metrics dictionary
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm
}

# Save to JSON
with open('./reports/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Evaluation complete. Metrics saved to ./reports/metrics.json")
print(metrics)
print()

# In eval.py, add:
vectorizer = joblib.load('./models/vectorizer.joblib')
ghana_news = "John Mahama Ghana Election 2024: NDC Massive Voter Fraud Uncovered - GhanaWeb"
ghana_news_stemmed = stemming(ghana_news)
ghana_news_tfidf = vectorizer.transform([ghana_news_stemmed])
prediction = model.predict(ghana_news_tfidf)
print("Ghanaian News Prediction:", "Real" if prediction[0] == 0 else "Fake")
print()