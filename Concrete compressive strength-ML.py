import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('concrete_data.csv')

df.drop_duplicates()
print(df.describe())
print(df)
print(df.isnull().sum())




# Load dataset


# Rename columns for easier use
df.columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 
              'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Compressive Strength']

# Data Overview
print(df.head())


# Data visualization
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Splitting data into features and target variable
X = df.drop(columns=["Compressive Strength"])
y = df["Compressive Strength"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model creation - RandomForest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plotting actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.title("Actual vs Predicted Compressive Strength")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

var = np.var(y_test,ddof = 0)
# Creating a DataFrame for actual and predicted values
results_df = pd.DataFrame({
    'Actual Compressive Strength': y_test,
    'Predicted Compressive Strength': y_pred
})

# Saving to an Excel file
results_df.to_excel('actual_vs_predicted_concrete_strength.xlsx', index=False)

print("File saved successfully as 'actual_vs_predicted_concrete_strength.xlsx'")
from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Randomized Search with cross-validation
rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                               n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train_scaled, y_train)

# Best parameters from Randomized Search
print(f"Best Parameters: {rf_random.best_params_}")

# Evaluate with the best model
best_model = rf_random.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

# Calculate new metrics with the best model
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)


print(f"Improved Mean Squared Error: {mse_best}")
print(f"Improved Root Mean Squared Error: {rmse_best}")
from sklearn.model_selection import cross_val_score

# Perform k-fold cross-validation (k=5)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculate mean of cross-validated MSE
cv_mse = -cv_scores.mean()
cv_rmse = np.sqrt(cv_mse)

print(f"Cross-validated Mean Squared Error: {cv_mse}")
print(f"Cross-validated Root Mean Squared Error: {cv_rmse}")
from sklearn.ensemble import GradientBoostingRegressor

# Initialize and fit Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_gb = gb_model.predict(X_test_scaled)

# Calculate new metrics with the Gradient Boosting model
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)


print(f"Gradient Boosting - Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting - Root Mean Squared Error: {rmse_gb}")

param_grid2 = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}
rf_random = RandomizedSearchCV(estimator=gb_model, param_distributions=param_grid2, 
                               n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train_scaled, y_train)

# Best parameters from Randomized Search
print(f"Best Parameters: {rf_random.best_params_}")

# Evaluate with the best model
best_model = rf_random.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

# Calculate new metrics with the best model
mse_best_gb = mean_squared_error(y_test, y_pred_best)
rmse_best_gb = np.sqrt(mse_best_gb)


print(f"Improved Mean Squared Error GB: {mse_best_gb}")
print(f"Improved Root Mean Squared Error GB: {rmse_best_gb}")
# ACCURACY SCORE #
r2_score = 1-(mse_best_gb/var)

print(f"Accuracy:{r2_score}")
results2_df = pd.DataFrame({
    'Actual Compressive Strength': y_test,
    'Predicted Compressive Strength': y_pred_best
})

# Saving to an Excel file
results2_df.to_excel('actual_vs_predicted_concrete_strength using Gradient Boosting.xlsx', index=False)

print("File saved successfully as 'actual_vs_predicted_concrete_strength using Gradient Boosting.xlsx'")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_gb, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.title("Actual vs Predicted Compressive Strength")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32,activation = 'relu'),
    Dense(1)
])

# Compile the model
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = nn_model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# Evaluate the model
y_pred_nn = nn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_nn)
print(f'Mean Squared Error: {mse}')

# Save the results
results3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_nn.flatten()})
results3.to_excel('predictions NN.xlsx', index=False)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_nn, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.title("Actual vs Predicted Compressive Strength for Neural Network")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
#MODEL COMPARISION
R_df = pd.DataFrame({
    'Actual': y_test,
    'NN_Predicted': y_pred_nn.flatten(),
    'RF_Predicted': y_pred_best
})

R_df.to_excel('model_comparison.xlsx', index=False)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot summary plot
shap.summary_plot(shap_values, X_test, feature_names=df.drop('Compressive Strength', axis=1).columns)

# Plot dependence plot for a specific feature
shap.dependence_plot('Cement', shap_values, X_test, feature_names=df.drop('Compressive Strength', axis=1).columns)
shap.dependence_plot('Coarse Aggregate', shap_values, X_test, feature_names=df.drop('Compressive Strength', axis=1).columns)
shap.dependence_plot('Fine Aggregate', shap_values, X_test, feature_names=df.drop('Compressive Strength', axis=1).columns)
shap.dependence_plot('Water', shap_values, X_test, feature_names=df.drop('Compressive Strength', axis=1).columns)
from sklearn.feature_selection import RFE


# Initialize and fit RFE
rfe = RFE(estimator=model, n_features_to_select=5)
rfe = rfe.fit(X, y)
# Get feature ranking
ranking = rfe.ranking_
features = X.columns

# Create a DataFrame to view feature importance
feature_ranking_df = pd.DataFrame({
    'Feature': features,
    'Ranking': ranking
}).sort_values(by='Ranking')

print("Feature Ranking:")
print(feature_ranking_df)

# Transform data to include only selected features
X_selected = X.loc[:, rfe.support_]
print("\nSelected Features DataFrame:")
print(X_selected.head())
feature_ranking = feature_ranking_df.to_excel("Ranking of features",index=False)
Selected_Features = X_selected.to_excel("Selected Features acc to importance",index=False)
#----------------------------------THANK YOU---------------------------------------------#
