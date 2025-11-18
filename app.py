import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
import sys
import os

# --- Import Preprocessing and Training Functions ---
# Assuming the provided code is saved as 'model.py'
# We need to import the specific functions and variables used for prediction
try:
    from model import preprocess_text
    # Note: We don't import train_and_save_model() to keep the web app clean, 
    # but we'll run it once to ensure the model and vectorizer exist.
except ImportError:
    print("❌ Error: Could not import 'preprocess_text' from 'model.py'. "
          "Please ensure the file is named 'model.py' and is in the same directory.")
    sys.exit(1)


# --- Flask Application Setup ---
app = Flask(__name__)

# Global variables to hold the loaded model and vectorizer
MODEL = None
VECTORIZER = None


def load_model_and_vectorizer():
    """Loads the pre-trained model and vectorizer."""
    global MODEL, VECTORIZER
    try:
        MODEL = joblib.load('model.pkl')
        VECTORIZER = joblib.load('vectorizer.pkl')
        print("✅ Model and Vectorizer loaded successfully.")
        return True
    except FileNotFoundError:
        print("⚠️ Warning: 'model.pkl' or 'vectorizer.pkl' not found.")
        print("Attempting to train the model now...")
        # Since the model wasn't found, try to train it
        from model import train_and_save_model
        if train_and_save_model():
            # If training was successful, try loading again
            return load_model_and_vectorizer()
        else:
            print("❌ Critical Error: Training failed or data files are missing. Cannot run prediction.")
            return False


# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the main prediction form and handles prediction on POST request."""
    prediction_result = None

    if request.method == 'POST':
        article_text = request.form.get('article_text')

        if not article_text:
            prediction_result = {"status": "error", "message": "Please enter some news text."}
        elif MODEL is None or VECTORIZER is None:
            prediction_result = {"status": "error", "message": "Model not loaded. Check server logs."}
        else:
            # 1. Preprocess the input text
            processed_text = preprocess_text(article_text)
            
            # 2. Vectorize the processed text
            text_vector = VECTORIZER.transform([processed_text])
            
            # 3. Predict the label
            prediction = MODEL.predict(text_vector)[0]
            
            # 4. Determine result text
            result_text = "TRUE" if prediction == 1 else "FAKE"
            
            prediction_result = {
                "status": "success",
                "text": article_text,
                "result": result_text,
                "label": int(prediction)
            }

    return render_template('project.html', prediction_result=prediction_result)

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure the model and vectorizer are loaded before running the app
    if load_model_and_vectorizer():
        print("🚀 Starting Flask app...")
        # Running on a standard development port
        app.run(debug=True)
    else:
        print("🛑 Application stopped due to critical error.")