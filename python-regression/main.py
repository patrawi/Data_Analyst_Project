import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def simple_linear_regression(df):
    y = np.array(df['NMHC(GT)']).reshape(-1,1)
    X = np.array(df['C6H6(GT)']).reshape(-1,1)
    X_trained, X_test, y_trained, y_test = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=42)
    lnr = LinearRegression()

    lnr.fit(X_trained, y_trained)
    b = lnr.intercept_[0]
    m = lnr.coef_[0][0]
    y_pred = lnr.predict(X_test)
    y_trained_pred = lnr.predict(X_trained)
    print("Mean Absolute Error Trained ", mean_absolute_error(y_trained, y_trained_pred))
    print("Mean Absolute Error Test ", mean_absolute_error(y_test, y_pred))
    scatterred_y = np.array(y_pred).reshape(-1)
    scatterred_x = np.array(X_test).reshape(-1)

    plt.axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')

    plt.scatter(scatterred_x, scatterred_y)
    plt.title('Simple Linear Regression')
    plt.xlabel('C6H6(GT)')
    plt.ylabel('NMHC(GT)')
    plt.show()

    return

def multiple_linear_regression(df):
    y = np.array(df['NMHC(GT)']).reshape(-1,1)
    df_pollutant = df[['PT08.S2(NMHC)', 'C6H6(GT)','T', 'RH', 'AH', 'NO2(GT)', 'NOx(GT)', 'CO(GT)']]
    X = np.array(df_pollutant).reshape(-1, len(df_pollutant.columns))
    X_trained, X_test, y_trained, y_test = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=42)
    lnr = LinearRegression()
    lnr.fit(X_trained, y_trained)
    y_pred = lnr.predict(X_test)
    y_trained_pred = lnr.predict(X_trained)
    print("Mean Absolute Error Trained ", mean_absolute_error(y_trained, y_trained_pred))
    print("Mean Absolute Error Test ", mean_absolute_error(y_test, y_pred))


    return

def main():

    df = pd.read_csv(filepath_or_buffer='AirQualityUCI.csv', sep=',', decimal=',', header=0)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df = df.replace(-200,np.nan).dropna()

    user_choice = input("Choose between simple and multiple linear regression: ")
    if user_choice == 'simple':
        simple_linear_regression(df)
    elif user_choice == 'multiple':
        multiple_linear_regression(df)
    else:
        raise Exception("Sorry, Invalid Choice")




    return

if __name__ == '__main__':
    main()
