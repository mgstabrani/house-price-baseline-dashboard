import streamlit as st
import importlib
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_rumah.csv", index_col=0).reset_index()

module = importlib.import_module('pages.1_Regresi')
input_user = module.input_user

Q1_harga=df.Harga.quantile(0.25)
Q3_harga=df.Harga.quantile(0.75)
IQR_harga=Q3_harga-Q1_harga
Lower_Whisker_harga = Q1_harga-1.5*IQR_harga
Upper_Whisker_harga = Q3_harga+1.5*IQR_harga
df = df[df.Harga < Upper_Whisker_harga]
Q1_luas_tanah=df["Luas Tanah"].quantile(0.25)
Q3_luas_tanah=df["Luas Tanah"].quantile(0.75)
IQR_luas_tanah=Q3_luas_tanah-Q1_luas_tanah
Lower_Whisker_luas_tanah = Q1_luas_tanah-1.5*IQR_luas_tanah
Upper_Whisker_luas_tanah = Q3_luas_tanah+1.5*IQR_luas_tanah
lower_luas_tanah = df["Luas Tanah"] < Lower_Whisker_luas_tanah
upper_luas_tanah = df["Luas Tanah"] > Upper_Whisker_luas_tanah
df = df[df["Luas Tanah"] < Upper_Whisker_luas_tanah]

wcss_list= [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(df[['Luas Tanah','Harga']])
    wcss_list.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(df[['Luas Tanah','Harga']])
df['cluster'] = y_predict
df1 = df[df['cluster'] == 0]
df2 = df[df['cluster'] == 1]
df3 = df[df['cluster'] == 2]
df4 = df[df['cluster'] == 3]
df5 = df[df['cluster'] == 4]


st.set_page_config(layout='wide')
st.title('Clustering Luas Tanah')

df_user = input_user()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df_user)
st.write('---')

st.write('---')
st.header('Data Visualisation')

# Scatter plot of KMeans clustering Luas Tanah dan Harga
st.subheader('Scatter plot of KMeans clustering Luas Tanah dan Harga')
fig = plt.figure()
plt.scatter(df1['Luas Tanah'], df1.Harga, color = 'green', label = 'Cluster 1')
plt.scatter(df2['Luas Tanah'], df2.Harga, color = 'red', label = 'Cluster 2')
plt.scatter(df3['Luas Tanah'], df3.Harga, color = 'black', label = 'Cluster 3')
plt.scatter(df4['Luas Tanah'], df4.Harga, color = 'blue', label = 'Cluster 4')
plt.scatter(df5['Luas Tanah'], df5.Harga, color = 'purple', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.xlabel('Luas Tanah')
plt.ylabel("Harga")
plt.legend()
st.pyplot(fig)