import pandas as pd

df = pd.read_csv("Practical\\jan20\\el.csv")

def percentf():
    docno = df.groupby("Profession").sum().loc["Doctor"]["Sno."]
    engno = df.groupby("Profession").sum().loc["Engineer"]["Sno."]
    tot = df.groupby("Profession").sum()["Sno."].sum()

    print(f"Percentage of Engineers: {round(((engno/tot)*100), 2)}%")
    print(f"Percentage of Doctors: {round(((docno/tot)*100), 2)}%")


def compare_add():
    tot = df["Salary"].sum()
    df["Salary Percentage"] = round(((df["Salary"]/tot)*100), 2)
    print(df.sort_values(by="Salary", ascending=True))

compare_add()

