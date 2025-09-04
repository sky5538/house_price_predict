import streamlit as st

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏡 Malaysia House Price Prediction App")

st.markdown("""
Welcome!  
You may use this app to predict house price.  
Some details you must enter to get a more accurate prediction:
1. Price per square feet  
2. Area  
3. State  
4. Tenure  
5. Type  
""")
st.markdown("""
Please train the model before predicting the house price.  
The Train page is on the sidebar. Three algorithms are used:
1. Random Forest  
2. Linear Regression  
3. Gradient Boosting
""")
st.markdown("""
Please note that the price is not 100% accurate and may differ slightly from the actual price.
""")


