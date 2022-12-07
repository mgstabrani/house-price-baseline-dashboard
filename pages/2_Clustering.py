import streamlit as st
import importlib
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

df_clean = pd.read_csv("cleaned_rumah.csv", index_col=0).reset_index()

df_dbscan = df_clean.copy()
df_dbscan = df_dbscan[(df_dbscan.Latitude < -4) & (df_dbscan.Longitude > 80)]
db =  DBSCAN(eps=0.05, min_samples=10)
df_dbscan["dbscan_cluster"] = db.fit_predict(df_dbscan[["Latitude","Longitude"]])

module = importlib.import_module('pages.1_Regresi')
input_user = module.input_user

st.set_page_config(layout='wide')
st.title('Clustering')

df = input_user()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

st.write('---')
st.header('Data Visualisation')

# Scatter plot of DBSCAN clustering
st.subheader('Scatter plot of DBSCAN clustering')
fig = plt.figure()
sns.scatterplot(data=df_dbscan, x="Latitude", y="Longitude", hue="dbscan_cluster", palette="deep")
st.pyplot(fig)

# Scatter plot of KMeans clustering
df_kmeans = df_dbscan[(df_dbscan['dbscan_cluster'] == 0)].copy()
kmeans = KMeans(n_clusters=5, random_state=42).fit(df_kmeans[["Latitude","Longitude"]])
df_kmeans["kmeans_cluster"] = kmeans.predict(df_kmeans[["Latitude","Longitude"]])
st.subheader('Scatter plot of KMeans clustering')
fig1 = plt.figure()
sns.scatterplot(data=df_kmeans, x="Latitude", y="Longitude", hue="kmeans_cluster", palette="deep")
st.pyplot(fig1)

# Scatter plot of KMeans between Latitude, Longitude, and Price (Harga)
df_plot = df_kmeans.copy()
df_plot = df_plot[df_plot['Harga'] > 0]
st.subheader('Scatter plot of KMeans between Latitude, Longitude, and Price (Harga)')
fig2 = plt.figure()
sns.scatterplot(data=df_plot, x="Latitude", y="Longitude", hue="Harga")
st.pyplot(fig2)