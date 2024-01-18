import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the Boston Housing Prices dataset
data = load_boston()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'KNN': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

# Initialize variables to keep track of the best model and its performance
best_model_name = None
best_mse = float('inf')

# Train and evaluate each model
for model_name, model in models.items():
    if model_name == 'KNN':
        # Try different values of k for KNN and select the best one using cross-validation
        best_k = None
        best_k_mse = float('inf')
        for k in range(1, 21):  # You can adjust the range of k as needed
            knn_model = KNeighborsRegressor(n_neighbors=k)
            mse_scores = -cross_val_score(knn_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            mean_mse = np.mean(mse_scores)
            if mean_mse < best_k_mse:
                best_k_mse = mean_mse
                best_k = k
        
        print(f"Best k for KNN: {best_k}")
        model = KNeighborsRegressor(n_neighbors=best_k)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{model_name} - Mean Squared Error: {mse}")

    
    if mse < best_mse:
        best_model_name = model_name
        best_mse = mse

print(f"The best model is {best_model_name} with MSE: {best_mse}")
