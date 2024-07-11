from edaClass import EDA

import pandas as pd
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Model(EDA):

    def __init__(self, df):
        super().__init__(df)

    def seperate_data(self, targetColumn):
        X = self.df.drop(targetColumn, axis=1)
        y = self.df[targetColumn]
        return X, y

    def train_knn_model(self, xTrain, xTest, yTrain, yTest):
        n_neighbors = st.sidebar.slider("Number of neighbors (K):", 1, 10)
        knnModel = KNeighborsClassifier(n_neighbors=n_neighbors)
        knnModel.fit(xTrain, yTrain)
        yPred = knnModel.predict(xTest)
        accuracy = accuracy_score(yTest, yPred)
        st.write(f"Accuracy of KNN Model: {round(accuracy*100,2)}")

    
    def train_svm_model(self, xTrain, xTest, yTrain, yTest):
        c = st.sidebar.slider("Regularization coefficient (C):", 0.01, 5.0)
        svmModel = SVC(C=c)
        svmModel.fit(xTrain, yTrain)
        yPred = svmModel.predict(xTest)
        accuracy = accuracy_score(yTest, yPred)
        st.write(f"Accuracy of SVM Model: {round(accuracy*100,2)}")
