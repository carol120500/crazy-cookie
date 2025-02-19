import pandas as pd
import shap
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
df = pd.read_csv('3month_dataset.csv')

# Select relevant features and target variable
features = ['Tide Level', 'Wind Speed', 'Atmospheric Pressure', 'Temperature Air',
            'Temperature Acqua', 'Cumulative Rainfall', 'Solar Radiation',
            'Relative Humidity', 'Water Level 1 Hour Ago']
target = 'Water Level'

# Drop rows with missing values
df.dropna(subset=features + [target], inplace=True)

# Step 2: Split the data into training and test sets
X = df[features]
y = df[target]

# Standardize the features (important for linear models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Reduce the background data using k-means clustering (K = 50)
K = 50  # Set the number of clusters for background data reduction
background = shap.kmeans(X_train, K)

# Create the SHAP explainer with the summarized background data
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test)  # Calculate SHAP values for the test set

# Step 5: Visualize the SHAP values

# Summary plot (shows feature importance)
shap.summary_plot(shap_values, X_test, feature_names=features)

# SHAP dependence plot for a specific feature (e.g., 'Tide Level')
shap.dependence_plot('Tide Level', shap_values, X_test, feature_names=features)

# Optional: Visualize SHAP values for individual predictions
# This shows how the model prediction was influenced by each feature
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test[0], feature_names=features)
