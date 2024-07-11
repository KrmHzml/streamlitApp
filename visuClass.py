import matplotlib.pyplot as plt
import seaborn as sns
from edaClass import EDA
import pandas as pd

class Visu(EDA):
    def __init__(self, df):
        super().__init__(df)

    def display_heatmap(self):
        fig, ax = plt.subplots()
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        return fig
    
    def display_pairplot(self):
        fig = sns.pairplot(self.df, diag_kind="kde")
        return fig
    
    def display_piechart(self):
        fig, ax = plt.subplots()
        self.df.value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('')
        return fig