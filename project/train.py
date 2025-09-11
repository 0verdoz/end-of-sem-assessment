import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import os

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Load preprocessed data
x_train = np.load('./data/x_train.npy', allow_pickle=True)
y_train = np.load('./data/y_train.npy', allow_pickle=True)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Save model
joblib.dump(model, './models/logistic_regression_model.joblib')

print("Training complete. Model saved to ./models/logistic_regression_model.joblib")