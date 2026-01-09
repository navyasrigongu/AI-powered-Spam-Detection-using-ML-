import streamlit as st
from model import predict_spam

st.title("AI Spam Detector")

msg = st.text_area("Enter Message")

if st.button("Check"):
    st.write(predict_spam(msg))
    #streamlit run app.py
