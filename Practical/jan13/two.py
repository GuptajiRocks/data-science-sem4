import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Practical\\jan13\\dp.csv")

def imputing_mean(df):
    imp_obj = si(strategy='mean')
    imda = imp_obj.fit_transform(df)
    return imda

def q1():
    print(df)
    imputing_mean()
    print(df.isna().sum().sum())

def std_sc(df):
    scl = StandardScaler()
    newd = scl.fit_transform(df)
    print(pd.DataFrame(newd))

    
def label_cat():
    label_encoder = LabelEncoder()
    df['Online Shopper'] = label_encoder.fit_transform(df['Online Shopper'])
    df["Region"] = label_encoder.fit_transform(df["Region"])
    df['Region_Binary'] = df['Region'].apply(lambda x: f'{x:03b}')
    df[['Brazil', 'India', 'USA']] = df['Region_Binary'].apply(lambda x: pd.Series(list(x)))
    df[['Brazil', 'India', 'USA']] = df[['Brazil', 'India', 'USA']].astype(int)
    return df

newdf = label_cat()
print(newdf)

imdf = imputing_mean(newdf)
print(pd.DataFrame(imdf))

std_sc(pd.DataFrame(imdf))
