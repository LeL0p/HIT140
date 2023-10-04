import statsmodels.api as sm

import numpy as np
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def MOTOR_UPDRS_model():
    # Read dataset into a DataFrame
    df = pd.read_csv("po2_data.csv")

    #replace any missing data with NAN
    df = df.replace('', np.nan)

    # Drop one or more of the correlated variables. Keep only one.
    df = df.drop(["shimmer(apq3)"], axis=1)
    df = df.drop(["shimmer(dda)"], axis=1)

    # print(df.info()) # For if you want to see the correct rows dropped away

    # Separate explanatory variables and response variables
    x = df.iloc[:, 6:].values

    y_motor = df.iloc[:, 4].values
    y_total = df.iloc[:, 5].values


    # Train-test split (60% training, 40% testing)
    x_train, x_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(x, y_motor, y_total, test_size=0.5, random_state=0)

    # Train simple linear regression models for motor_updrs and total_updrs
    motor_model = LinearRegression()


    # Build and evaluate the linear regression model
    x = sm.add_constant(x)
    model = sm.OLS(y_motor,x).fit()
    pred = model.predict(x)
    model_details = model.summary()
    print(model_details)

    # Train (fit) the linear regression model using the training set
    motor_model.fit(x_train, y_motor_train)


    # Print the intercept and coefficient learned by the linear regression model
    print("Intercept: ", motor_model.intercept_)
    print("Coefficient: ", motor_model.coef_)

    # Predictions
    y_motor_pred = motor_model.predict(x_test)

    # Optional: Show the predicted values of (y) next to the actual values of (y) for motor_updrs
    df_pred_motor = pd.DataFrame({"Actual Motor ": y_motor_test, "Predicted Motor ": y_motor_pred})
    print("Predictions for Motor UPDRS:")
    print(df_pred_motor)
    # print(df_pred_motor.head(10))  # Print the first few rows for visualization

    # Compute standard performance metrics of the linear regression:
    # Calculate mean absolute error
    motor_mae = metrics.mean_absolute_error(y_motor_test, y_motor_pred)

    # Calculate mean squared error
    motor_mse = metrics.mean_squared_error(y_motor_test, y_motor_pred)

    # Calculate root mean squared error
    motor_rmse = math.sqrt(motor_mse)

    # Calculate normalized root mean squared error (NRMSE)
    y_motor_max = y_motor_test.max()
    y_motor_min = y_motor_test.min()
    motor_nrmse_norm = motor_rmse / (y_motor_max - y_motor_min)

    # Calculate coefficient of determination (R-squared)
    motor_r_2 = metrics.r2_score(y_motor_test, y_motor_pred)

    # Calculate adjusted coefficient of determination
    n_motor = len(y_motor_test)  # number of samples for motor_updrs
    p = x_test.shape[1]  # number of rows in x variables
    motor_adj_r_2 = 1 - ((1 - motor_r_2) * (n_motor - 1) / (n_motor - p - 1))

    # Print the metrics
    print("\nMetrics for Motor UPDRS:")
    print("Mean Absolute Error:", motor_mae)
    print("Mean Squared Error:", motor_mse)
    print("Root Mean Squared Error:", motor_rmse)
    print("Normalized RMSE:", motor_nrmse_norm)
    print("R-squared (coefficient of determination):", motor_r_2)
    print("Adjusted R-squared:", motor_adj_r_2)



def TOTAL_UPDRS_model():
    # Read dataset into a DataFrame
    df = pd.read_csv("po2_data.csv")

    #replace any missing data with NAN
    df = df.replace('', np.nan)

    # Drop one or more of the correlated variables. Keep only one.
    df = df.drop(["shimmer(apq3)"], axis=1)
    df = df.drop(["shimmer(dda)"], axis=1)
    df = df.drop(["jitter(ppq5)"], axis=1)

    # print(df.info()) # For if you want to see the correct rows dropped away
    


    # Separate explanatory variables and response variables
    x = df.iloc[:, 6:].values

    #merge the entries for each individual
    merged_row1 = df.iloc[2:151].astype(float).mean() 
    mean_values = pd.DataFrame(merged_row1).T 

    merged_row2 = df.iloc[151:296].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row2]

    merged_row3 = df.iloc[296:440].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row3]

    merged_row4 = df.iloc[440:577].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row4]

    merged_row5 = df.iloc[577:733].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row5]

    merged_row6 = df.iloc[733:889].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row6]

    merged_row7 = df.iloc[889:1050].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row7]

    merged_row8 = df.iloc[1050:1200].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row8]

    merged_row9 = df.iloc[1200:1352].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row9]

    merged_row10 = df.iloc[1352:1500].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row10]

    merged_row11 = df.iloc[1500:1638].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row11]

    merged_row12 = df.iloc[1638:1745].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row12]

    merged_row13 = df.iloc[1745:1857].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row13]

    merged_row14 = df.iloc[1857:1993].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row14]

    merged_row15 = df.iloc[1993:2136].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row15]

    merged_row16 = df.iloc[2136:2274].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row16]

    merged_row17 = df.iloc[2274:2418].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row17]

    merged_row18 = df.iloc[2418:2544].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row18]

    merged_row19 = df.iloc[2544:2673].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row19]

    merged_row20 = df.iloc[2673:2807].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row20]

    merged_row21 = df.iloc[2807:2930].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row21]

    merged_row22 = df.iloc[2930:3042].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row22]

    merged_row23 = df.iloc[3042:3180].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row23]

    merged_row24 = df.iloc[3180:3336].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row24]

    merged_row25 = df.iloc[3336:3480].astype(float).mean()
    mean_values = df.loc[len(mean_values)] = [merged_row25]

    merged_row26 = df.iloc[3480:3610].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row26]

    merged_row27 = df.iloc[3610:3739].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row27]

    merged_row28 = df.iloc[3739:3873].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row28]

    merged_row29 = df.iloc[3873:4041].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row29]

    merged_row30 = df.iloc[4041:4167].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row30]

    merged_row31 = df.iloc[4167:4297].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row31]

    merged_row32 = df.iloc[4297:4398].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row32]

    merged_row33 = df.iloc[4398:4533].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row33]

    merged_row34 = df.iloc[4533:4694].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row34]

    merged_row35 = df.iloc[4694:4859].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row35]

    merged_row36 = df.iloc[4859:4988].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row36]

    merged_row37 = df.iloc[4988:5128].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row37]

    merged_row38 = df.iloc[5128:5277].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row38]

    merged_row39 = df.iloc[5277:5420].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row39]

    merged_row40 = df.iloc[5420:5562].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row40]

    merged_row41 = df.iloc[5562:5727].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row41]

    merged_row42 = df.iloc[5727:5877].astype(float).mean() 
    mean_values = df.loc[len(mean_values)] = [merged_row42]


    y_motor = df.iloc[:, 4].values
    y_total = df.iloc[:, 5].values

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

    print("\nMetrics for Total UPDRS:")
    print("Mean Absolute Error:", total_mae)
    print("Mean Squared Error:", total_mse)
    print("Root Mean Squared Error:", total_rmse)
    print("Normalized RMSE:", total_nrmse_norm)
    print("R-squared (coefficient of determination):", total_r_2)
    print("Adjusted R-squared:", total_adj_r_2)
    print (mean_values)


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