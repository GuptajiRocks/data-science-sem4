import pandas as pd

df = pd.read_excel("Lectures\\24apr-self\\jesus.xlsx")

newdf = df[["Registration No", "Student Name", "Percentage"]]

def percent_75():
    print(newdf[newdf["Percentage"] > 75])
    print(len(newdf[newdf["Percentage"] > 75]))

def avl():
    print(newdf[(newdf["Percentage"] > 60) & (newdf["Percentage"] < 75)])

def cooked():
    print(newdf[newdf["Percentage"] < 60])

#avl()

cooked()