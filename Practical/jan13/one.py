import pandas as pd
from sklearn import preprocessing as pt

# For normalization import minmaxscaler/standard scaler

# For imputation simple imputer

# For converting discrete to continous, we do Label Encoding.

df = pd.read_csv("Practical\\jan13\\main.csv")
print(df)

def normalize(df):
    mn = pt.MinMaxScaler(feature_range=(0,1))
    newdf = mn.fit_transform(df)
    return newdf

ndf = normalize(df)

def print_normalise():
    print(ndf)
    print(pd.DataFrame(ndf))


def binarize(df):
    thresh = 0.0
    binas = pt.Binarizer(threshold=thresh)
    bina_df = binas.fit_transform(df)
    return bina_df

bdf = binarize(df)

def print_binarize():
    print(bdf)
    print(pd.DataFrame(bdf))

def standard_gauss(df):
    stds = pt.StandardScaler()
    sdata = stds.fit_transform(df)
    return sdata

sdf = standard_gauss(df)

def print_gauss():
    print(sdf)
    print(pd.DataFrame(sdf))

print_gauss
print_binarize
print_gauss()