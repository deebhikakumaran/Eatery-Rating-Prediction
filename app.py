import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load trained model 
# @st.cache_resource
# def load_model():
#     model = pickle.load(open("rating_model.pkl", "rb"))
#     vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
#     return model, vectorizer

# model, vectorizer = load_model()

# Load the pickled pipeline and store map
try:
    # with open('final_pipeline.pkl', 'rb') as f:
    #     pipeline = pickle.load(f)

    with open('store_map.pkl', 'rb') as f:
        store_map = pickle.load(f)

except FileNotFoundError:
    st.error("Model files not found. Please run the `pickle_model.py` script first.")
    st.stop()

# Streamlit UI
st.title('McDonald\'s Rating Prediction')

st.markdown("""
This application predicts the star rating (1-5) for a McDonald's review based on the selected branch, review text, and review count.
""")

# User inputs
branch_name = st.selectbox('Select Branch Name',options=list(store_map.keys()))
rating_count = st.number_input("Rating Count", min_value=0, value=0)
# review_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
review_text = st.text_area('Write your review here')
# review_time = st.date_input("Review Date")
review_time = '5 years ago'

if st.button('Predict Rating'):
    if not review_text:
        st.warning("Please write a review before predicting.")
    else:
        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'store_name': [branch_name],
            'rating_count': [rating_count],
            'review_time': [review_time],
            'review': [review_text]
        })

        # Preprocess the data and make a prediction
        try:
            # prediction = pipeline.predict(input_data)[0]
            st.success(f'The predicted rating for this review is: stars ⭐')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")






