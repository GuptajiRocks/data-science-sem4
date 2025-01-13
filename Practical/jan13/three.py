import pandas as pd
import sklearn.preprocessing as st
import sklearn.impute as si

df = pd.read_csv("Practical\\jan13\\el.csv")

def imputing():
    obj = si.SimpleImputer(strategy='mean')
    df[["Salary", "Age"]] = obj.fit_transform(df[["Salary", "Age"]])
    print(df)
    
imputing()
    
