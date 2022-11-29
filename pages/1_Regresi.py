from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout='wide')
st.title('Regresi')

cleaned_data = pd.read_csv("cleaned_rumah.csv", index_col=[0])
cleaned_data.head()

cleaned_data.shape

cleaned_data.info()

cleaned_data.describe()

cleaned_data.columns = cleaned_data.columns.str.lower().str.replace(" ", "_")
cleaned_data.rename(columns={'harga_per_m^2': 'harga_tanah'}, inplace=True)

cleaned_data['harga'] = cleaned_data['harga'].apply(lambda x: x/1000000)

cleaned_data = cleaned_data.drop(["alamat", "jenis_interior"], axis=1)
cleaned_data.head()

cleaned_data = cleaned_data[cleaned_data['kamar_tidur'] > 0]
cleaned_data = cleaned_data[cleaned_data['kamar_mandi'] > 0]
cleaned_data = cleaned_data[cleaned_data['luas_bangunan'] >= 36]
cleaned_data = cleaned_data[cleaned_data['luas_tanah'] >= 36]
cleaned_data = cleaned_data[cleaned_data['listrik'] >= 450]
cleaned_data = cleaned_data[cleaned_data['harga_tanah'] >= 1000000]
cleaned_data = cleaned_data.drop_duplicates()


def boxplot(df_new, name):
    sns.boxplot(df_new[name])
    plt.title("Box Plot before median imputation")
    plt.show()
    q1 = df_new[name].quantile(0.25)
    q3 = df_new[name].quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    med = np.median(df_new[name])
    for i in df_new[name]:
        if i > Upper_tail or i < Lower_tail:
            df_new[name] = df_new[name].replace(i, med)
    sns.boxplot(df_new[name])
    plt.title("Box Plot after median imputation")
    plt.show()


def outlier(df, name):
    Q1 = df[name].quantile(0.25)
    Q3 = df[name].quantile(0.75)
    IQR = Q3 - Q1
    df = df.query('(@Q1 - 1.5 * @IQR) <=' + name + '<= (@Q3 + 1.5 * @IQR)')
    return df


numeric_col = cleaned_data.select_dtypes(include=np.number).columns.tolist()
numeric_col = numeric_col[:-4]

cleaned_data = outlier(cleaned_data, "harga")

df_corr = cleaned_data.corr()

cleaned_data = cleaned_data.drop(["latitude", "longitude"], axis=1)

X = cleaned_data.drop('harga', axis=1)
y = cleaned_data['harga']

sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

rfr = RandomForestRegressor(n_estimators=40)
rfr_algo = make_pipeline(sc, rfr)

rfr_algo.fit(X_train, y_train)

rfr_pred_train = rfr_algo.predict(X_train)
rfr_pred_test = rfr_algo.predict(X_test)

st.title("Prediksi Harga Rumah")
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

st.sidebar.title("Input User")
st.sidebar.write("Masukkan data rumah yang ingin diprediksi")

# Input User


def input_user():
    bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", int(cleaned_data["kamar_tidur"].min(
    )), int(cleaned_data["kamar_tidur"].max()), 1)
    bathrooms = st.sidebar.slider("Jumlah Kamar Mandi", int(cleaned_data["kamar_mandi"].min(
    )), int(cleaned_data["kamar_mandi"].max()), 1)
    sqft_living = st.sidebar.slider(
        "Luas Tanah", float(cleaned_data["luas_tanah"].min()), float(cleaned_data["luas_tanah"].max()), float(cleaned_data["luas_tanah"].mean()))
    sqft_lot = st.sidebar.slider("Luas Bangunan", float(cleaned_data["luas_bangunan"].min(
    )), float(cleaned_data["luas_bangunan"].max()), float(cleaned_data["luas_bangunan"].mean()))
    price = st.sidebar.slider(
        "Harga tanah (m²)", float(cleaned_data["harga"].min()), float(cleaned_data["harga"].max()), float(cleaned_data["harga"].mean()))
    parking = st.sidebar.slider("Jumlah Tempat Parkir", int(cleaned_data["tempat_parkir"].min(
    )), int(cleaned_data["tempat_parkir"].max()), 1)
    watt = st.sidebar.slider("Daya Listrik", float(cleaned_data["listrik"].min()), float(
        cleaned_data["listrik"].max()), float(cleaned_data["listrik"].mean()))

    interior_options = ["interior_lengkap", "interior_multiple_options_available",
                        "interior_sebagian", "interior_tak_berperabot"]

    interior = st.sidebar.selectbox("Interior", interior_options)

    if interior == "interior_lengkap":
        interior_lengkap = 1
        interior_multiple_options_available = 0
        interior_sebagian = 0
        interior_tak_berperabot = 0
    elif interior == "interior_multiple_options_available":
        interior_lengkap = 0
        interior_multiple_options_available = 1
        interior_sebagian = 0
        interior_tak_berperabot = 0
    elif interior == "interior_sebagian":
        interior_lengkap = 0
        interior_multiple_options_available = 0
        interior_sebagian = 1
        interior_tak_berperabot = 0
    else:
        interior_lengkap = 0
        interior_multiple_options_available = 0
        interior_sebagian = 0
        interior_tak_berperabot = 1

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'price': price,
        'parking': parking,
        'watt': watt,
        'interior_lengkap': interior_lengkap,
        'interior_multiple_options_available': interior_multiple_options_available,
        'interior_sebagian': interior_sebagian,
        'interior_tak_berperabot': interior_tak_berperabot
    }

    features = pd.DataFrame(data, index=[0])
    return features


df = input_user()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Apply model to make predictions
prediction = rfr_algo.predict(df)
st.header('Prediction of House Price')
st.write(prediction)
st.write('---')

st.write('---')
st.header('Data Visualisation')
st.write('---')

# Scatter predictions vs actual
st.subheader('Scatter plot of predictions vs actual')
fig = plt.figure()
plt.scatter(y_test, rfr_pred_test)
plt.plot([0, 1], [0, 1], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
st.pyplot(fig)

# Histogram between prediction and actual
st.subheader('Histogram of predictions vs actual')
fig2 = plt.figure()
plt.hist(y_test, bins=20, alpha=0.5, label='Actual')
plt.hist(rfr_pred_test, bins=20, alpha=0.5, label='Predicted')
plt.xlabel('House Price')
plt.ylabel('Count')
plt.legend(loc='upper right')
st.pyplot(fig2)

# Barplot between prediction and actual
st.subheader('Barplot of predictions vs actual')
fig3 = plt.figure()
plt.bar(y_test, rfr_pred_test)
plt.xlabel('Actual')
plt.ylabel('Predicted')
st.pyplot(fig3)

# Scatter Plot
st.subheader('Scatter Plot')
fig5 = plt.figure()
plt.scatter(cleaned_data['luas_tanah'], cleaned_data['harga'])
plt.xlabel('Luas Tanah')
plt.ylabel('Harga')
st.pyplot(fig5)

# Bar Plot
st.subheader('Bar Plot')
fig7 = plt.figure()
sns.barplot(x='kamar_tidur', y='harga', data=cleaned_data)
st.pyplot(fig7)

# Bar Plot
fig8 = plt.figure()
sns.barplot(x='kamar_mandi', y='harga', data=cleaned_data)
st.pyplot(fig8)
