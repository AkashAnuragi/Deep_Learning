# Importing Libraries and Utilies
import streamlit as st
import joblib
from utils import text_clean
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")

st.set_page_config(page_title="Review Analysis" , layout='wide')

st.header("Review Analysis Positive/Negative")
st.text("Write Your Review For The Hotel, Hotels and their services are amazing but we want to know your view/opnion")

review = st.text_area("Write Your Review Here...")
if st.button("Submit"):
    review = text_clean(review)
    data = tfidf.transform([review])
    pred = model.predict(data)
    if pred[0]==1:
        st.success("Positive Review!")
        st.warning("Thank You For Your Feedback!")
    else:
        st.error("Negative Review!")
        st.warning("Thank You For Your Feedback!")
        st.success("We are sorry for inconvenience! We Will Take care of it next time!")
    