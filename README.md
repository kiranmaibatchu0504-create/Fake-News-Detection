 Fake News Detection

A simple Streamlit app to detect fake news using a Logistic Regression model trained on text data.

## Features
- Input news text to predict if it's Real or Fake.
- Model trained using scikit-learn's Logistic Regression.

## How to Run
1. Clone the repo: `git clone <repo-url>`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Model Details
- Uses TF-IDF vectorization + Logistic Regression.
- Model files: `lr_model.joblib`, `vectorizer.joblib`
