import pandas as pd

class EDA:
    def __init__(self, df):
        self.df = df

    def display_shape(self):
        return self.df.shape
    
    def display_columns(self):
        return self.df.columns
    
    def display_dataframe(self):
        return self.df.head(10)
    
    def display_summary(self):
        return self.df.describe()
    
    def display_null_values(self):
        return self.df.isnull().sum()
    
    def display_datatypes(self):
        return self.df.dtypes

