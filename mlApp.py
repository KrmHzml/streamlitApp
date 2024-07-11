from PIL import Image

from edaClass import EDA
from visuClass import Visu
from modelClass import Model

from sklearn.model_selection import train_test_split

import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

st.title("Full ML Apps")

img = Image.open("Machine_Learning.jpg")
img = img.resize((1000,500), Image.Resampling.LANCZOS)
st.image(img)


def main():
    activities = ["EDA", "Visualization", "Model", "About me"]
    option = st.sidebar.selectbox("Select option: ", activities)

    if option == "EDA":
        st.subheader("Exploratory Data Analysis")
        
        data = st.file_uploader("Upload dataset: ", type=["csv"])
        if data is not None:
            st.success("Data succesfully loaded")
            df = pd.read_csv(data)
        
            eda = EDA(df)
            st.write(eda.display_dataframe())
            

            if st.checkbox("Display shape"):
                st.write(eda.display_shape())
            
            if st.checkbox("Display columns"):
                st.write(eda.display_columns())

            if st.checkbox("Select multiple columns"):
                selectedColumns = []
                selectedColumns = st.multiselect("Select preferred columns: ", df.columns)
                eda1 = EDA(df[selectedColumns])
                st.write(eda1.display_dataframe())

            if st.checkbox("Display summary"):
                st.write(eda.display_summary())

            if st.checkbox("Display Null Values"):
                st.write(eda.display_null_values())

            if st.checkbox("Display data types"):
                st.write(eda.display_datatypes())

    elif option == "Visualization":
        st.subheader("Visualization")
        
        data = st.file_uploader("Upload dataset: ", type=["csv"])
        if data is not None:
            st.success("Data succesfully loaded")
            df = pd.read_csv(data)
        
            visu = Visu(df)
            st.write(visu.display_dataframe())

            if st.text("Select multiple columns to plot"):
                selectedColumns = []
                selectedColumns = st.multiselect("", visu.display_columns())
                visu1 = Visu(df[selectedColumns])

            if st.checkbox("Display heatmap correlation"):
                fig = visu1.display_heatmap()
                st.pyplot(fig)    

            if st.checkbox("Display pairplot"):
                fig = visu1.display_pairplot()
                st.pyplot(fig)
                
            if st.checkbox("Display pie chart"):
                fig = visu1.display_piechart()
                st.pyplot(fig)     

    elif option == "Model":
        st.subheader("Model")
        
        data = st.file_uploader("Upload dataset: ", type=["csv"])
        if data is not None:
            st.success("Data succesfully loaded")
            df = pd.read_csv(data)

            selectedColumns = []
            selectedColumns = st.multiselect("Select columns",df.columns)

            df = df[selectedColumns]
            model = Model(df)
            st.write(model.display_dataframe())


            selectedModel = st.sidebar.selectbox("Select Model", ["KNN", "SVM"])

            targetColumn = st.selectbox("Select target column", model.display_columns())

            X, y = model.seperate_data(targetColumn)
            st.write("X values:", X.head())
            st.write("y values:", y.head())

            xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

            if selectedModel == "KNN":
                model.train_knn_model(xTrain, xTest, yTrain, yTest)
            elif selectedModel == "SVM":
                model.train_svm_model(xTrain, xTest, yTrain, yTest)

    elif option == "About me":
        st.write("Designed by Kerem Hüzmeli")

            


            

if __name__ == "__main__":
    main()