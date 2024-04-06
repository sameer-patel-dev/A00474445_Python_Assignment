import streamlit as st
from single_crypto_analysis import show_page as show_single_crypto_analysis
from crypto_comparison import show_page as show_crypto_comparison
from digit_model import show_page as show_digit_model


st.sidebar.title("Python Assignment - MCDA")
page = st.sidebar.radio("Please navigate", ["Stock Details", "Stock Comparison", "Digit Classifier"])

if page == "Stock Details":
    show_single_crypto_analysis()

elif page == "Stock Comparison":
    show_crypto_comparison()

else:
    show_digit_model()

