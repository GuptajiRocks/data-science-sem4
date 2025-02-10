import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Practical\\feb10\\cpa.xls")

def qa():
    x = df["horsepower"]
    y = df["citympg"]
    x = sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    pred = model.predict(x)
    sns.scatterplot(x=df["horsepower"], y=df["citympg"], label="Actual", color="blue")
    sns.lineplot(x=df["horsepower"], y=pred, label="Best Fit Line", color="red")
    plt.show()
    residuals = model.resid
    sns.histplot(residuals, kde=True)
    plt.show()

def qb():
    x = df["horsepower"]
    y = df["highwaympg"]
    x = sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    print(model.summary())
    pred = model.predict(x)
    sns.scatterplot(x=df["horsepower"], y=df["highwaympg"], label="Actual", color="blue")
    sns.lineplot(x=df["horsepower"], y=pred, label="Best Fit Line", color="red")
    plt.show()
    residuals = model.resid
    sns.histplot(residuals, kde=True)
    plt.show()

def qc():
    x1 = df[["citympg"]]
    x2= df[["highwaympg"]]
    y = df["price"]
    # x1 = sm.add_constant(x1)
    # x2 = sm.add_constant(x2)
    # model1 = sm.OLS(y,x1).fit()
    # model2 = sm.OLS(y, x2).fit()
    model1 = LinearRegression()
    model2 = LinearRegression()
    model1.fit(x1, y)
    model2.fit(x2, y)
    pred1 = model1.predict(x1)
    pred2 = model2.predict(x2)
    print(f"R-squared of Model1 {model1.score(x1, y)}")
    print(f"R-squared of Model 2 {model2.score(x2, y)}")
    sns.scatterplot(x=df["citympg"], y=df["price"], label="CityMPG vs Price", color="blue")
    sns.scatterplot(x=df["highwaympg"], y=df["price"], label="HighwayMPG vs Price", color="magenta")
    sns.lineplot(x=df["citympg"], y=pred1, label="CityMPG", color="red")
    sns.lineplot(x=df["highwaympg"], y=pred2, label="Highway MPG", color="orange")
    plt.show()

qc()