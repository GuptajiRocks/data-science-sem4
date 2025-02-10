import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

olddf = pd.read_csv("Practical\\feb10\\auto-mpg.csv")
olddf = olddf.dropna()

df = olddf[["cylinders", "horsepower", "mpg","acceleration"]]


X = df[['mpg', 'cylinders']]
y = df['acceleration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train['mpg'], y=y_train, label="Training Data", color="blue", alpha=0.6)
sns.scatterplot(x=X_test['mpg'], y=y_test, label="Testing Data", color="red", alpha=0.6)
plt.xlabel("MPG")
plt.ylabel("Acceleration")
plt.title("MPG vs Acceleration")
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_train['cylinders'], y=y_train, label="Training Data", color="green", alpha=0.6)
sns.scatterplot(x=X_test['cylinders'], y=y_test, label="Testing Data", color="orange", alpha=0.6)
plt.xlabel("Cylinders")
plt.ylabel("Acceleration")
plt.title("Cylinders vs Acceleration")
plt.legend()

plt.tight_layout()
plt.show()
