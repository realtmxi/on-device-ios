import coremltools as ct
import joblib

# Load the trained model
model = joblib.load('naive_bayes_model.joblib')

import json

model_params = {
    "class_log_prior": model.class_log_prior_.tolist(),
    "feature_log_prob": model.feature_log_prob_.tolist(),
    "classes": model.classes_.tolist()
}

with open('naive_bayes_params.json', 'w') as f:
    json.dump(model_params, f)


