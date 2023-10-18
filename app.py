import streamlit as st
import joblib
import json 
import os
from collections import Counter

rf = joblib.load(os.path.join('files','rf.sav'))
knn = joblib.load(os.path.join('files','knn.sav'))
lr = joblib.load(os.path.join('files','lr.sav'))
dt = joblib.load(os.path.join('files','dt.sav'))
mnb = joblib.load(os.path.join('files','nb.sav'))
vect = joblib.load(os.path.join('files','vect.sav'))
with open(os.path.join('files','stopwords.json'),'r') as fp:
    stopwords = json.load(fp)


def preprocess_descp(descp):
    descp = ' '.join([word for word in descp.split(' ') if word not in list(stopwords.values())])
    arr = vect.transform([descp])
    return arr  
    
def model_prediction(model,arr):
    prob = model.predict_proba(arr).tolist()[0]
    if prob[0] > prob[1]:
        prob = prob[0]
    else:
        prob = prob[1]
    pred = model.predict(arr)[0]  
    return round(prob,2),pred

def main():
    st.title("Fake Job Predictor App")

    # Input fields
    user = st.text_input("Enter Username")
    job_description = st.text_area("Enter Job Description")

    if st.button("Predict"):
        if user and job_description:
            arr = preprocess_descp(job_description)
            
            rf_prob,rf_pred = model_prediction(rf,arr)
            st.markdown(f"**Random Forest Prediction (best model): {rf_pred,rf_prob}**")
            
            mnb_prob,mnb_pred = model_prediction(mnb,arr)
            st.markdown(f"**Naive Bayes Prediction: {mnb_pred,mnb_prob}**")
            
            dt_prob,dt_pred = model_prediction(dt,arr)
            st.markdown(f"**Decision Tree Prediction: {dt_pred,dt_prob}**")
            
            knn_prob,knn_pred = model_prediction(knn,arr)
            st.markdown(f"**KNN Prediction: {knn_pred,knn_prob}**")
            
            lr_prob,lr_pred = model_prediction(lr,arr)
            st.markdown(f"**Logistic Regression Prediction: {lr_pred,lr_prob}**")
            
            # final_pred = Counter([rf_pred,mnb_pred,knn_pred,lr_pred,dt_pred]).most_common(1)[0][0]
            # final_prob = (rf_prob+mnb_prob+dt_prob+knn_prob+lr_prob)/5
            # st.markdown(f"**Cumulative Prediction: {final_pred,round(final_prob,2)}**")
        else:
            st.warning("Please enter both Username and Job Description.")

if __name__ == "__main__":
    main()
