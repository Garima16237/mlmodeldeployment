import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor
#from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder 
import openpyxl


st.sidebar.image("compunnel.jpg",width=200)
st.sidebar.header("Food Cost Prediction")
result = st.sidebar.radio(label ='Click on buttons',options=['Data information', 'Correlation heatmap','Prediction','Scatterplot'])

if result == 'Data information':
    st.header('Data Information')
    data = pd.read_excel('Retention_CP_prediction.xlsx',index_col=0) 
    st.write(data)

elif result == 'Correlation heatmap':
    st.header('Correlation Heatmap')
    data = pd.read_excel("Retention_CP_prediction.xlsx",index_col=0)
    fig, ax = plt.subplots(figsize=(20,15))
    sns.heatmap(data=data.corr() ,annot = True)
    st.write(fig)

elif result == 'Prediction':
    st.header('Predicted Values')
    data = pd.read_excel("Retention_CP_prediction.xlsx",index_col=0)
    encoder = OneHotEncoder()
    final_data = encoder.fit_transform(data.drop(columns='cost'))
    X = final_data
    Y= data['cost']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    model = ExtraTreeRegressor(random_state=42)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    st.table(predictions) 
    
    

else :
    st.header('Scatter Plot Of Predicted Values')
    data = pd.read_excel("Retention_CP_prediction.xlsx",index_col=0)
    encoder = OneHotEncoder()
    final_data = encoder.fit_transform(data.drop(columns='cost'))
    X = final_data
    Y = data['cost']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    model = ExtraTreeRegressor(random_state=42)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)

    plt.scatter(Y_test,predictions)

    st.write(fig)