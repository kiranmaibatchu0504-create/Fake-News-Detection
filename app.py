import streamlit as st
import joblib
import os

# Define the path to your vectorizer file
VECTORIZER_PATH = "vectorizer.jb"  # Update this path if necessary

# Load the vectorizer with error handling using st.cache_data
@st.cache_data
def load_vectorizer():
    if not os.path.exists(VECTORIZER_PATH):
        st.error(f"Vectorizer file not found at '{VECTORIZER_PATH}'. Please upload or place it in the correct directory.")
        return None
    try:
        return joblib.load(VECTORIZER_PATH)
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        return None

vectorizer = load_vectorizer()

# Your app logic here
def main():
    st.title("Fake News Detection")
    
    if vectorizer is None:
        st.stop()
    
    user_input = st.text_area("Enter news text to analyze:")
    
    if st.button("Analyze"):
        # Example: convert text to features
        features = vectorizer.transform([user_input])
        # Your prediction code here
        # For example:
        # prediction = model.predict(features)
        # st.write(f"Prediction: {prediction}")
        st.write("This is where your prediction result will appear.")

if __name__ == "__main__":
    main()
