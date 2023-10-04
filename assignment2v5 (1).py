import statsmodels.api as sm

import numpy as np
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("po2_data.csv")

#replace any missing data with NAN
df = df.replace('', np.nan)

# Drop one or more of the correlated variables. Keep only one.
df = df.drop(["shimmer(apq3)"], axis=1)
df = df.drop(["shimmer(dda)"], axis=1)

# print(df.info()) # For if you want to see the correct columns dropped away

# Separate explanatory variables and response variables
x = df.iloc[:, 6:].values
y_motor = df.iloc[:, 4].values
y_total = df.iloc[:, 5].values



# Train-test split (60% training, 40% testing)
x_train, x_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(x, y_motor, y_total, test_size=0.5, random_state=0)

# Train simple linear regression models for motor_updrs and total_updrs
motor_model = LinearRegression()
total_model = LinearRegression()

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y_motor,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

model2 = sm.OLS(y_total,x).fit()
pred2 = model2.predict(x)
model_details2 = model2.summary()
print(model_details2)

# Train (fit) the linear regression model using the training set
motor_model.fit(x_train, y_motor_train)
total_model.fit(x_train, y_total_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", motor_model.intercept_)
print("Coefficient: ", motor_model.coef_)

print("Intercept: ", total_model.intercept_)
print("Coefficient: ", total_model.coef_)

# Predictions
y_motor_pred = motor_model.predict(x_test)
y_total_pred = total_model.predict(x_test)

# Optional: Show the predicted values of (y) next to the actual values of (y) for motor_updrs
df_pred_motor = pd.DataFrame({"Actual Motor ": y_motor_test, "Predicted Motor ": y_motor_pred})
print("Predictions for Motor UPDRS:")
print(df_pred_motor)
# print(df_pred_motor.head(10))  # Print the first few rows for visualization

# Optional: Show the predicted values of (y) next to the actual values of (y) for total_updrs
df_pred_total = pd.DataFrame({"Actual Total ": y_total_test, "Predicted Total ": y_total_pred})
print("\nPredictions for Total UPDRS:")
print(df_pred_total)
# print(df_pred_total.head(10))  # Print the first few rows for visualization


# Compute standard performance metrics of the linear regression:
# Calculate mean absolute error
motor_mae = metrics.mean_absolute_error(y_motor_test, y_motor_pred)
total_mae = metrics.mean_absolute_error(y_total_test, y_total_pred)

# Calculate mean squared error
motor_mse = metrics.mean_squared_error(y_motor_test, y_motor_pred)
total_mse = metrics.mean_squared_error(y_total_test, y_total_pred)

# Calculate root mean squared error
motor_rmse = math.sqrt(motor_mse)
total_rmse = math.sqrt(total_mse)

# Calculate normalized root mean squared error (NRMSE)
y_motor_max = y_motor_test.max()
y_motor_min = y_motor_test.min()

y_total_max = y_total_test.max()
y_total_min = y_total_test.min()

motor_nrmse_norm = motor_rmse / (y_motor_max - y_motor_min)
total_nrmse_norm = total_rmse / (y_total_max - y_total_min)


# Calculate coefficient of determination (R-squared)
motor_r_2 = metrics.r2_score(y_motor_test, y_motor_pred)
total_r_2 = metrics.r2_score(y_total_test, y_total_pred)


# Calculate adjusted coefficient of determination
n_motor = len(y_motor_test)  # number of samples for motor_updrs
n_total = len(y_total_test)  # number of samples for total_updrs
p = x_test.shape[1]  # number of columns in x variables
motor_adj_r_2 = 1 - ((1 - motor_r_2) * (n_motor - 1) / (n_motor - p - 1))
total_adj_r_2 = 1 - ((1 - total_r_2) * (n_total - 1) / (n_total - p - 1))

# Print the metrics
print("\nMetrics for Motor UPDRS:")
print("Mean Absolute Error:", motor_mae)
print("Mean Squared Error:", motor_mse)
print("Root Mean Squared Error:", motor_rmse)
print("Normalized RMSE:", motor_nrmse_norm)
print("R-squared (coefficient of determination):", motor_r_2)
print("Adjusted R-squared:", motor_adj_r_2)

print("\nMetrics for Total UPDRS:")
print("Mean Absolute Error:", total_mae)
print("Mean Squared Error:", total_mse)
print("Root Mean Squared Error:", total_rmse)
print("Normalized RMSE:", total_nrmse_norm)
print("R-squared (coefficient of determination):", total_r_2)
print("Adjusted R-squared:", total_adj_r_2)