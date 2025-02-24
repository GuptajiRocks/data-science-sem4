import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assuming your data is in a DataFrame called 'df'
# If you need to load the data, uncomment and modify the following line:
df = pd.read_csv('Practical\\feb22\\creditcard.csv')

# a. Select features (V1-V28) and visualize using V1 and V2 as example
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Class'] == 0]['V1'], 
           df[df['Class'] == 0]['V2'], 
           color='blue', label='Normal', alpha=0.6)
plt.scatter(df[df['Class'] == 1]['V1'], 
           df[df['Class'] == 1]['V2'], 
           color='red', label='Fraud', alpha=0.6)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Credit Card Transactions: Normal vs Fraud')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# b. Divide features and labels
feature_columns = [f'V{i}' for i in range(1, 29)]  # V1 to V28
X = df[feature_columns].values
y = df['Class'].values

# c. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# d. Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# e. Fit KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# f. Predict on test set
y_pred = knn.predict(X_test_scaled)

# g. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Additional visualization of decision boundaries (using V1 and V2)
def plot_decision_boundary(X, y, model, scaler, feature_idx=[0, 1]):
    h = 0.02  # step size in the mesh
    
    # Create mesh grid using only the two selected features
    x_min, x_max = X[:, feature_idx[0]].min() - 1, X[:, feature_idx[0]].max() + 1
    y_min, y_max = X[:, feature_idx[1]].min() - 1, X[:, feature_idx[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create the full feature space with zeros for unused features
    mesh_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
    mesh_points[:, feature_idx[0]] = xx.ravel()
    mesh_points[:, feature_idx[1]] = yy.ravel()
    
    # Scale the mesh points
    mesh_points = scaler.transform(mesh_points)
    
    # Predict for each point in the mesh
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, feature_idx[0]], X[:, feature_idx[1]], c=y, alpha=0.8)
    plt.xlabel('V1')
    plt.ylabel('V2')
    plt.title('KNN Decision Boundary (V1 vs V2)')
    plt.show()

# Plot decision boundary using V1 and V2
plot_decision_boundary(X, y, knn, scaler, feature_idx=[0, 1])

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# try:
#     df = pd.read_csv('Practical\\feb22\\creditcard.csv')
# except FileNotFoundError:
#     print("Error: 'creditcard_fraud.csv' not found. Please ensure the file exists in the correct directory.")
#     exit()

# plt.figure(figsize=(10, 6))
# plt.scatter(df['distance_from_home'][df['fraud'] == 0], df['ratio_to_median_purchase_price'][df['fraud'] == 0], color='blue', label='Normal')
# plt.scatter(df['distance_from_home'][df['fraud'] == 1], df['ratio_to_median_purchase_price'][df['fraud'] == 1], color='red', label='Fraud')
# plt.xlabel('distance_from_home')
# plt.ylabel('ratio_to_median_purchase_price')
# plt.title('Credit Card Fraud Detection')
# plt.legend()
# plt.show()

# X = df[['distance_from_home', 'ratio_to_median_purchase_price']]
# y = df['fraud']

# # c. Split the train and test into 80:20 ratio.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # d. Normalize the x_train and x_test using StandardScaler.
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # e. Fit KNeighborsClassifier on x_train and y_train.
# knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors
# knn.fit(X_train_scaled, y_train)

# # f. Predict x_test.
# y_pred = knn.predict(X_test_scaled)

# # g. Calculate accuracy using y_pred and y_test.
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')