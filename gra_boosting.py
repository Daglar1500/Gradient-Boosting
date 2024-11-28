import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Step 1: Load the dataset from the uploaded file
file_path = 'winequality-red.csv'
df = pd.read_csv(file_path)

# Step 2: Inspect the data to ensure proper formatting
print("Dataset Head:")
print(df.head())

# Step 3: Define features (X) and target (y)
X = df.drop(columns=["quality"])  # Features
y = df["quality"]                # Target variable

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train a Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=100,  # Number of trees
    learning_rate=0.1, # Step size
    max_depth=3,       # Tree depth
    random_state=42    # Reproducibility
)

model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# Step 8: Feature importance (optional)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)

# Scatter plot for actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="orange")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Ideal prediction line
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Quality")
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color="purple", alpha=0.7)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()

mse_list = []
for y_pred_stage in model.staged_predict(X_test):
    mse_list.append(mean_squared_error(y_test, y_pred_stage))

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(mse_list)), mse_list, marker='o', color="green")
plt.xlabel("Number of Trees")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve of Gradient Boosting")
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Gradient Boosting")
plt.gca().invert_yaxis()  # Display the most important feature at the top
plt.show()
