import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the vectorizer and model
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('naive_bayes_model.joblib')

def assess_risk(sentence):
    # Transform the input sentence to a feature vector
    X = vectorizer.transform([sentence])
    
    # Make a prediction
    prediction = model.predict(X)
    
    # Convert to boolean
    is_risk = bool(prediction[0])
    
    return is_risk

# Example sentence
sentence = "Please email me at murphy@xiaohongshu.com."
is_risk_detected = assess_risk(sentence)

print(f"Privacy Leak Risk Detected: {is_risk_detected}")
