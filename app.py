from pycaret.regression import load_model, predict_model
import streamlit as st
from readline import set_pre_input_hook
from sys import setprofile
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    


model = load_model('salud_model')


st.image('1615502068951.jpeg', use_column_width=False, width=500)
st.title('Medical power')
st.write('NOT FOR COMERCIAL USE, IS A TEST FOR WINE QUALITY.')

age = st.sidebar.slider(label = 'age', min_value = 1.0, max_value = 99.0 , value = 33.0, step = 1.0)
sex = st.sidebar.selectbox('Sex',('male','female'))
bmi = st.sidebar.slider(label = 'bmi', min_value = 1.0, max_value = 99.0 , value = 33.0, step = 1.0)
children = st.sidebar.slider(label = 'children', min_value = 1.0, max_value = 99.0 , value = 33.0, step = 1.0)
smoker = st.sidebar.selectbox('smoker',('yes','no'))
region = st.sidebar.selectbox('region',('southeast','northeast','northwest','southwest'))


features = {"age":age, "sex":sex, "bmi":bmi, "children":children, "smoker":smoker, "region":region}


df = pd.DataFrame(features, index = [0])
prediction = predict_quality(model, df)       
features_df  = pd.DataFrame([features])




st.table(features_df.T)

if st.button('Predict'):    
    prediction = predict_quality(model, features_df)    
    st.write('The corretly charge evaluating the risk is  '+ str(prediction))


