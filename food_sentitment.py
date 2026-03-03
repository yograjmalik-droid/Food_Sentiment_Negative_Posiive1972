import streamlit as st
import joblib
import pandas as pd
import numpy as np

vectorizer=joblib.load("vectorizer.pkl")
model=joblib.load("sentiment_model.pkl")

st.set_page_config(layout="wide")
st.sidebar.image("myphoto.jpeg")
st.sidebar.title("About Project")
st.sidebar.write("The goal of this project is to perform sentiment analysis on food reviews to determine whether they are positive or negative")
st.sidebar.title("Libraries")
st.sidebar.markdown("""
- Pandas → Load and clean food review data
- NumPy → Handle numerical computations
- Sklearn → Train model to predict Positive/Negative sentiment
""")

st.sidebar.title("Cloud-based Streamlit application")
st.sidebar.markdown("A Streamlit application deployed on the cloud. A cloud-hosted application built using Streamlit. A Streamlit web application running on a cloud platform. A cloud-enabled Streamlit app for online access")


st.sidebar.title("Contact")
st.sidebar.markdown("📞9990576610 YOGRAJ MALIK")
st.markdown("""
<style>
.banner {
    background-color: #00B280;
    padding: 25px;
    border-radius: 10px;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
</style>
<div class="banner">
Food Sentiment Analysis
</div>
""", unsafe_allow_html=True)
st.write("\n")
col1,col2=st.columns([.4,.6])
with col1:
    st.header("Predict Single Review")
    review=st.text_input("**Enter Review**")
    if st.button("Predict"):
        X_test=vectorizer.transform([review])
        pred=model.predict(X_test)
        prob=model.predict_proba(X_test)
        if pred[0]==0:
            st.error("**Sentiment = Negative👎**")
            st.warning(f"Confidance Score = {prob[0][0]:.2f}")
        else:
            st.success("**Sentiemnt = Positive 👍**")
            st.warning(f"Confidance Score {prob[0][1]:.2f}")
with col2:
    st.header("Predict Bulk Reviews from CSV")
    file=st.file_uploader("**Select a csv file**",type=["csv","txt"])
    if file:
        df=pd.read_csv(file,header=None,names=["Review"])
        placeholder = st.empty()
        placeholder.dataframe(df)
        if st.button("Bulk Prediction"):
            X_test=vectorizer.transform(df.Review)
            pred=model.predict(X_test)
            prob=model.predict_proba(X_test)
            sentiment=["Positive" if i==1 else "Negative" for i in pred]
            df['Sentiment']=sentiment
            df['Confidence']=np.max(prob,axis=1)
            placeholder.dataframe(df)
            

