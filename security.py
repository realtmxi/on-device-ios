import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
violations_df = pd.read_csv('gdpr_violations.csv')
text_df = pd.read_csv('gdpr_text.csv')

# Extract positive and negative samples
positive_samples = violations_df['summary']
negative_samples = text_df['gdpr_text']

# Label the data
positive_labels = [1] * len(positive_samples)
negative_labels = [0] * len(negative_samples)

X = pd.concat([positive_samples, negative_samples], ignore_index=True)
y = positive_labels + negative_labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Precision: {precision_score(y_test, y_pred):.2f}')
print(f'Recall: {recall_score(y_test, y_pred):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.2f}')

# Export the model
import joblib
joblib.dump(model, 'naive_bayes_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
