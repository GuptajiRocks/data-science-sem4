import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as si

df = pd.read_csv("Practical\\jan13\\main.csv")

def imputing_mean():
    imp_obj = si(strategy='mean')
    imda = imp_obj.fit_transform(df)
    # print(imda)
    print(pd.DataFrame(imda))
    return imda

def q1():
    print(df)
    imputing_mean()

    print(df.isna().sum().sum())

def std_sc():
    scl = StandardScaler()
    newd = scl.fit_transform(imputing_mean())
    print(newd)

std_sc()
