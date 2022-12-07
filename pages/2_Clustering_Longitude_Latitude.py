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

df_kmeans = df_dbscan[(df_dbscan['dbscan_cluster'] == 0)].copy()
kmeans = KMeans(n_clusters=5, random_state=42).fit(df_kmeans[["Latitude","Longitude"]])
df_kmeans["kmeans_cluster"] = kmeans.predict(df_kmeans[["Latitude","Longitude"]])

longitude = st.sidebar.slider("Longitude", float(df_clean["Longitude"].min(
    )), float(df_clean["Longitude"].max()), 0.1)
latitude = st.sidebar.slider("Latitude", float(df_clean["Latitude"].min(
    )), float(df_clean["Latitude"].max()), 0.1)

st.title('Clustering Longitude and Latitude')

# Print specified input parameters
st.header('Specified Input parameters')
df_user = pd.DataFrame({
    'Longitude': longitude,
    'Latitude': latitude
}, index=[0])
st.write(df_user)
st.write('---')

# Apply model to make predictions
predictionKmeans = kmeans.predict(df_user)
predictionDBSCAN = db.fit_predict(df_user)
st.header('Prediction of House Cluster')
st.write(pd.DataFrame({
    'KMeans': predictionKmeans,
    'DBSCAN': predictionDBSCAN
}))
st.write('---')

st.header('Data Visualisation')

# Scatter plot of DBSCAN clustering
st.subheader('Scatter plot of DBSCAN clustering')
fig = plt.figure()
sns.scatterplot(data=df_dbscan, x="Latitude", y="Longitude", hue="dbscan_cluster", palette="deep")
st.pyplot(fig)

# Scatter plot of KMeans clustering
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