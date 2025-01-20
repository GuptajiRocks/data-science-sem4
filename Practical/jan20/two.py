from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("Practical\\jan20\\data.csv")

def parts():
    print(df)
    print(df.shape)
    print(df.head())
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    print(df.isna().sum())

unrem = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
#print(unrem.isna().sum())

X = unrem.drop(columns=["diagnosis"])
y = unrem["diagnosis"]

obj_scaler = StandardScaler()
finscal = obj_scaler.fit_transform(X)

pca = PCA(n_components = 2)
pcadone = pca.fit_transform(finscal)

pca_df = pd.DataFrame(pcadone, columns=["PC1", "PC2"])
pca_df["diagnosis"] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="diagnosis", palette="Set1", alpha=0.7)
plt.title("Scatter Plot of PCA-Reduced Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Diagnosis")
plt.show()
