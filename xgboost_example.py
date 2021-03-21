import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import train_test_split, standardize, to_categorical, normalize
from utils import mean_squared_error, accuracy_score
from xgboost_model import XGBoost

#https://zhuanlan.zhihu.com/p/32181687

def main():
    print("-- XGBoost --")

    # Load temperature data
    data = pd.read_csv('temperature.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T  # shape=(366,1)  numpy.ndarray
    temp = np.atleast_2d(data["temp"].values).T

    X = time.reshape((-1, 1))  # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)  # Insert bias term
    print(type(X), X.shape, X)
    y = temp[:, 0]  # Temperature. Reduce to one-dim

    print('=' * 100)
    print(type(y), y.shape, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,shuffle=True)
    # print(y_train)
    model = XGBoost()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(y_test[0:5])
    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print("Mean Squared Error:", mse)

    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
