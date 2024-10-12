import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the vectorizer and model
vectorizer = joblib.load('../src/vectorizer.joblib')
model = joblib.load('../src/naive_bayes_model.joblib')

def assess_risk(sentence):
    # Transform the input sentence to a feature vector
    X = vectorizer.transform([sentence])
    
    # Make a prediction
    prediction = model.predict(X)
    
    # Convert to boolean
    is_risk = bool(prediction[0])
    
    # Identify contributing words
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = X.nonzero()[1]
    
    contributing_words = [feature_names[index] for index in nonzero_indices]
    
    # Calculate the contribution score for each word
    word_contributions = {}
    for word in contributing_words:
        index = feature_names.tolist().index(word)
        # Use model coefficients to assess contribution if available
        contribution = model.feature_log_prob_[1][index]
        word_contributions[word] = contribution
    
    return is_risk, word_contributions

# Example sentence
sentence1 = "hello world"
sentence2 = "my email is tmx@gmail.com"
is_risk_detected, contributions = assess_risk(sentence1)

print(f"Privacy Leak Risk Detected: {is_risk_detected}")
print("Contributing words and their contributions:", contributions)

is_risk_detected, contributions = assess_risk(sentence2)

print(f"Privacy Leak Risk Detected: {is_risk_detected}")
print("Contributing words and their contributions:", contributions)
