# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Practical\\feb10\\auto-mpg.csv")

df = df[['mpg', 'acceleration', 'cylinders']].dropna()

X = df[['mpg', 'cylinders']]
y = df['acceleration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_mpg = LinearRegression()
model_mpg.fit(X_train[['mpg']], y_train)

model_cylinders = LinearRegression()
model_cylinders.fit(X_train[['cylinders']], y_train)

y_pred_mpg = model_mpg.predict(X_test[['mpg']])
y_pred_cylinders = model_cylinders.predict(X_test[['cylinders']])

rmse_mpg = np.sqrt(mean_squared_error(y_test, y_pred_mpg))
r2_mpg = r2_score(y_test, y_pred_mpg)

rmse_cylinders = np.sqrt(mean_squared_error(y_test, y_pred_cylinders))
r2_cylinders = r2_score(y_test, y_pred_cylinders)

print(f"MPG Model -> R²: {r2_mpg:.4f}, RMSE: {rmse_mpg:.2f}")
print(f"Cylinders Model -> R²: {r2_cylinders:.4f}, RMSE: {rmse_cylinders:.2f}")


plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test['mpg'], y=y_test, label="Actual", color="blue", alpha=0.6)
sns.lineplot(x=X_test['mpg'], y=y_pred_mpg, label="Predicted (Best Fit Line)", color="red")
plt.xlabel("MPG")
plt.ylabel("Acceleration")
plt.title("MPG vs Acceleration (Regression)")
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test['cylinders'], y=y_test, label="Actual", color="green", alpha=0.6)
sns.lineplot(x=X_test['cylinders'], y=y_pred_cylinders, label="Predicted (Best Fit Line)", color="orange")
plt.xlabel("Cylinders")
plt.ylabel("Acceleration")
plt.title("Cylinders vs Acceleration (Regression)")
plt.legend()

plt.tight_layout()
plt.show()
