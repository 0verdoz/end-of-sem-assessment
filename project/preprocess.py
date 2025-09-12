import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

# Ensure directories exist
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)

def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

if __name__ == "__main__":
    # Load dataset
    news_dataset = pd.read_csv('./data/train.csv')  # Adjust path as needed
    news_dataset = news_dataset.fillna('')

    # Merge author and title
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

    # Apply stemming
    news_dataset['content'] = news_dataset['content'].apply(stemming)

    # Prepare X and y
    x = news_dataset['content'].values
    y = news_dataset['label'].values

    # Vectorize
    vectorizer = TfidfVectorizer()
    x_vec = vectorizer.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_vec, y, test_size=0.2, stratify=y, random_state=2
    )

    # Save preprocessed data and vectorizer
    np.save('./data/x_train.npy', x_train.toarray())
    np.save('./data/x_test.npy', x_test.toarray())
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
    joblib.dump(vectorizer, './models/vectorizer.joblib')

    print("Preprocessing complete. Data and vectorizer saved.")