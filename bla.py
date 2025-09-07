import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "resources/Concrete_Data.xls"
concrete_data = pd.read_excel(url)

# Display the first few rows and check for missing values
print(concrete_data.head())
print(f"Dataset shape: {concrete_data.shape}")
print(f"Missing values: {concrete_data.isnull().sum().sum()}")

#Let’s visualize the relationship between cement (a primary ingredient) and compressive strength:
plt.scatter(concrete_data.iloc[:, 0], concrete_data.iloc[:, -1])
plt.xlabel('Cement (kg/m³)')
plt.ylabel('Compressive Strength (MPa)')
plt.title('Cement vs. Compressive Strength')
plt.grid(True)
plt.show()

#We can also use Pandas to create a correlation matrix to identify relationships between variables:

# Calculate correlation matrix
correlation_matrix = concrete_data.corr()

# Display the correlation with the target variable
# Calculate correlation matrix
correlation_matrix = concrete_data.corr()

# Display the correlation with the target variable
print("Correlation with Compressive Strength:")
print(correlation_matrix.iloc[-1, :-1].sort_values(ascending=False))

#After exploring our dataset, let’s prepare it for machine learning by transforming our Pandas DataFrame
# into NumPy arrays suitable for scikit-learn models.

# Split the data into features (X) and target variable (y)
X = concrete_data.iloc[:, :-1]  # Features: All columns except the last
y = concrete_data.iloc[:, -1]  # Target: Only the last column

# Here we transition from Pandas to NumPy by converting DataFrames to arrays
# This is a key integration point between the two libraries
X_array = X.values  # Pandas DataFrame → NumPy array
y_array = y.values  # Pandas Series → NumPy array

print(f"Type before conversion: {type(X)}")
print(f"Type after conversion: {type(X_array)}")

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")

#Now let’s build and evaluate machine learning models using our processed data:
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
models = ["Linear Regression", "Random Forest"]
predictions = [lr_predictions, rf_predictions]

for model_name, pred in zip(models, predictions):
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{model_name}:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")

#Finally, let’s improve our model by incorporating domain knowledge about concrete:
# Use NumPy's efficient arithmetic operations to create domain-specific features
cement_water_ratio = X_train[:, 0] / X_train[:, 3]  # Cement / Water ratio
cement_water_ratio_test = X_test[:, 0] / X_test[:, 3]

# Add this new feature to our feature matrices using NumPy's array manipulation
X_train_enhanced = np.column_stack((X_train, cement_water_ratio))
X_test_enhanced = np.column_stack((X_test, cement_water_ratio_test))

# Train a model with the enhanced features
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_enhanced, y_train)
predictions = model.predict(X_test_enhanced)

print(f"Model with domain knowledge:")
print(f"  R² Score: {r2_score(y_test, predictions):.2f}")

# Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Strength (MPa)')
plt.ylabel('Predicted Strength (MPa)')
plt.title('Predicted vs Actual Concrete Strength')
plt.grid(True)
plt.show()