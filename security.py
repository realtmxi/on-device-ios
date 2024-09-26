import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
gdpr_violations_path = '~/Downloads/archive/gdpr_violations.csv'
data = pd.read_csv(gdpr_violations_path)

# Assuming 'summary' and 'article_violation' are your text and target features
X = data['summary']
y = data['article_violated'].apply(lambda x: 1 if pd.notnull(x) else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical data
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

import joblib

# Save the vectorizer and model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'naive_bayes_model.joblib')

