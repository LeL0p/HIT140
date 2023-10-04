import math

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Create global var of DF to call outside of function
df = pd.read_csv("po2_data.csv")
df = df.replace('', np.nan)

#Remove outliers to clean data with IQR method
def remove_outliers_iqr(df, column_name, lower_bound=0.25, upper_bound=0.75):
    q1 = df[column_name].quantile(lower_bound)
    q3 = df[column_name].quantile(upper_bound)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    df = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]

    return df

#Remove outliers with columns containing int
for column in df.columns:
    if df[column].dtype in [np.float64, np.int64]: 
        df = remove_outliers_iqr(df, column)

def MOTOR_UPDRS_model():
    # Read dataset into a DataFrame
    global df

    # Drop the correlated variables
    correlated_variables = ["shimmer(apq3)", "shimmer(dda)"]
    df = df.drop(correlated_variables, axis=1)

    # Separate explanatory variables and response variables
    X = df.iloc[:, 6:]
    y_motor = df.iloc[:, 4]

    # Train-test split (60% training, 40% testing)
    X_train, X_test, y_motor_train, y_motor_test = train_test_split(X, y_motor, test_size=0.4, random_state=0)

    # Train the linear regression model using statsmodels
    X_train = sm.add_constant(X_train)  # Add a constant (intercept) term
    model = sm.OLS(y_motor_train, X_train).fit()

    # Evaluate the linear regression model
    X_test = sm.add_constant(X_test)  # Add a constant (intercept) term to the test data
    y_motor_pred = model.predict(X_test)

    # Calculate the performance metrics
    motor_mae = metrics.mean_absolute_error(y_motor_test, y_motor_pred)
    motor_mse = metrics.mean_squared_error(y_motor_test, y_motor_pred)
    motor_rmse = math.sqrt(motor_mse)

    # Print the results
    print("MOTOR UPDRS Model Results:")
    print("Mean Absolute Error:", motor_mae)
    print("Mean Squared Error:", motor_mse)
    print("Root Mean Squared Error:", motor_rmse)

    # Plot the predicted values against the actual values
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_motor_test, y=y_motor_pred)
    plt.xlabel("Actual MOTOR UPDRS")
    plt.ylabel("Predicted MOTOR UPDRS")
    plt.title("Actual vs. Predicted MOTOR UPDRS")
    plt.plot([min(y_motor_test), max(y_motor_test)], [min(y_motor_test), max(y_motor_test)], linestyle='--', color='red')
    plt.show()
    
def TOTAL_UPDRS_model():
    # Read dataset into a DataFrame
    global df 

    # Drop one or more of the correlated variables. Keep only one.
    df = df.drop(["shimmer(apq3)"], axis=1)
    df = df.drop(["shimmer(dda)"], axis=1)
    df = df.drop(["jitter(ppq5)"], axis=1)

    # print(df.info()) # For if you want to see the correct rows dropped away
    
    # Group the data by 'subject number'
    grouped_data = df.groupby('subject#')

    # Calculate the mean for each column for each subject
    mean_by_subject = grouped_data.mean()

    #to make sure it works
    # print(mean_by_subject)

    # Separate explanatory variables and response variables
    x = mean_by_subject.iloc[:, 6:].values
    
    y_motor = mean_by_subject.iloc[:, 4].values
    y_total = mean_by_subject.iloc[:, 5].values

    # Train-test split (60% training, 40% testing)
    x_train, x_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(x, y_motor, y_total, test_size=0.5, random_state=0)

    # Train simple linear regression models for motor_updrs and total_updrs
    total_model = LinearRegression()

    # Build and evaluate the linear regression model
    model2 = sm.OLS(y_total,x).fit()
    pred2 = model2.predict(x)
    model_details2 = model2.summary()
    print(model_details2)

    # Train (fit) the linear regression model using the training set
    total_model.fit(x_train, y_total_train)

    # Print the intercept and coefficient learned by the linear regression model
    print("Intercept: ", total_model.intercept_)
    print("Coefficient: ", total_model.coef_)

    # Predictions
    y_total_pred = total_model.predict(x_test)

    # Optional: Show the predicted values of (y) next to the actual values of (y) for total_updrs
    df_pred_total = pd.DataFrame({"Actual Total ": y_total_test, "Predicted Total ": y_total_pred})
    print("\nPredictions for Total UPDRS:")
    print(df_pred_total)
    # print(df_pred_total.head(10))  # Print the first few rows for visualization

    # Compute standard performance metrics of the linear regression:
    # Calculate mean absolute error
    total_mae = metrics.mean_absolute_error(y_total_test, y_total_pred)

    # Calculate mean squared error
    total_mse = metrics.mean_squared_error(y_total_test, y_total_pred)

    # Calculate root mean squared error
    total_rmse = math.sqrt(total_mse)

    # Calculate normalized root mean squared error (NRMSE)
    y_total_max = y_total_test.max()
    y_total_min = y_total_test.min()
    total_nrmse_norm = total_rmse / (y_total_max - y_total_min)


    # Calculate coefficient of determination (R-squared)
    total_r_2 = metrics.r2_score(y_total_test, y_total_pred)

    # Calculate adjusted coefficient of determination
    n_total = len(y_total_test)  # number of samples for total_updrs
    p = x_test.shape[1]  # number of rows in x variables
    total_adj_r_2 = 1 - ((1 - total_r_2) * (n_total - 1) / (n_total - p - 1))

userchoice = 1

while userchoice != 0:
    userchoice = input("Input '1' to view prediction model for MOTOR UPDRS, '2' to view prediction model for TOTAL UPDRS, '3' for both and '0' to EXIT: ")
    if int(userchoice) == 1:
        MOTOR_UPDRS_model()
    elif int(userchoice) == 2:
        TOTAL_UPDRS_model()
    elif int(userchoice) == 3:
        MOTOR_UPDRS_model()
        TOTAL_UPDRS_model()
    else:
        print("Exit!")
        break