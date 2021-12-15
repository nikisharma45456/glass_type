import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



@st.cache()
def prediction(model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe):
  glass_type =model.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
  if glass_type ==1 :
    return "buildings windows float processed"
  elif  glass_type == 2 :
    return "building windows non float processed"
  elif  glass_type == 3:
    "vehicle windows float processed"
  elif glass_type == 4:
    "vehicle windows non float processed"
  elif glass_type == 5:
    "containers"
  elif glass_type == 6:
    "tableware"
  else:
    return "headlamp"

st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")
  
if st.sidebar.checkbox("Show Raw Data") :
  st.subheader("Full Dataset")
  st.dataframe(glass_df)

st.sidebar.subheader("Scatter Plot")
features_lists = st.sidebar.multiselect("select the x-axis",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

for i in features_lists(): 
  st.subheader(f"Scatter Plot between {i} and glasstype") 
  plt.figure(figsize= (20,4))
  sns.scatterplot(glass_df[i],glass_df["GlassType"])
  st.pyplot()

st.sidebar.subheader("Histogram")
features_lists = st.sidebar.multiselect("select the features fr histogram",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Choosing features for histograms.
for i in features_lists(): 
  st.subheader(f"Histogram for {i} ") 
  plt.figure(figsize= (20,4))
  plt.hist(glass_df[i],bins = "sturges")
  st.pyplot()

st.sidebar.subheader("Box Plot")
features_lists = st.sidebar.multiselect("select the features for box plot ",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Choosing columns for box plots.
for i in features_lists(): 
  st.subheader(f"Box Plot for {i} ") 
  plt.figure(figsize= (20,4))
  sns.boxplot(glass_df[i])
  st.pyplot()