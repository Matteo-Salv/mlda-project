from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model  import Ridge
import numpy as np
import matplotlib.pyplot as plt



class Regression:

    def __init__(self, dataset):
        x = dataset.drop(['charges'], axis=1)
        y = dataset.charges
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=0)

    def linearRegression(self):
        lr = LinearRegression().fit(self.x_train, self.y_train)
        y_train_pred = lr.predict(self.x_train)
        y_test_pred = lr.predict(self.x_test)
        print(lr.score(self.x_test, self.y_test))
        print("### RESULTS FOR LINEAR REGRESSION ###")
        print('MSE train data: %.3f, MSE test data: %.3f' %
              (metrics.mean_squared_error(y_train_pred, self.y_train),
               metrics.mean_squared_error(y_test_pred, self.y_test)))
        # plt.style.use('seaborn')
        # plt.scatter(y_train_pred, y_test_pred, color='red', marker='o', s=35, alpha=0.5,
        #             label='Test data')
        # plt.plot(y_train_pred, y_train_pred, color='blue', label='Model Plot')
        # plt.title('Predicted Values vs Inputs')
        # plt.xlabel('Inputs')
        # plt.ylabel('Predicted Values')
        # plt.legend(loc='upper left')
        # plt.show()

        plt.scatter(y_train_pred, self.y_train)
        plt.title('Linear Regression')
        tmp = [min(np.concatenate((y_train_pred, y_test_pred))),
               max(np.concatenate((y_train_pred, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()



    def randomForest(self):
        rfr = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
        rfr.fit(self.x_train, self.y_train)
        y_train_pred = rfr.predict(self.x_train)
        y_test_pred = rfr.predict(self.x_test)
        print("### RESULTS FOR RANDOM FOREST REGRESSION ###")
        print('MSE train data: %.3f, MSE test data: %.3f' %
              (metrics.mean_squared_error(y_train_pred, self.y_train),
               metrics.mean_squared_error(y_test_pred, self.y_test)))
        plt.figure(figsize=(8, 6))
        plt.title('Random Forest')
        plt.scatter(y_train_pred, y_train_pred - self.y_train,
                    c='gray', marker='o', s=35, alpha=0.5,
                    label='Train data')
        plt.scatter(y_test_pred, y_test_pred - self.y_test,
                    c='blue', marker='o', s=35, alpha=0.7,
                    label='Test data')
        plt.xlabel('Predicted values')
        plt.ylabel('Actual values')
        plt.legend(loc='upper right')
        plt.hlines(y=0, xmin=0, xmax=60000, lw=2, color='red')
        plt.show()


    def supportVectorRegression(self):
        svr = SVR(kernel='rbf')
        svr.fit(self.x_train, self.y_train)
        y_train_pred = svr.predict(self.x_train)
        y_test_pred = svr.predict(self.x_test)
        print("### RESULTS FOR SUPPORT VECTOR REGRESSION ###")
        print('MSE train data: %.3f, MSE test data: %.3f' %
              (metrics.mean_squared_error(y_train_pred, self.y_train),
               metrics.mean_squared_error(y_test_pred, self.y_test)))
        plt.title('SVR')
        plt.scatter(y_train_pred, self.y_train)
        tmp = [min(np.concatenate((y_train_pred, y_test_pred))),
               max(np.concatenate((y_train_pred, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()


    def KRLS(self):
        ridge = Ridge(alpha=0.5)
        ridge.fit(self.x_train, self.y_train)
        y_train_pred = ridge.predict(self.x_train)
        y_test_pred = ridge.predict(self.x_test)
        print("### RESULTS FOR KRLS ###")
        print('MSE train data: %.3f, MSE test data: %.3f' %
              (metrics.mean_squared_error(y_train_pred, self.y_train),
               metrics.mean_squared_error(y_test_pred, self.y_test)))
        plt.title('KRLS')
        plt.scatter(y_train_pred, self.y_train)
        tmp = [min(np.concatenate((y_train_pred, y_test_pred))),
               max(np.concatenate((y_train_pred, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()

        """
        TODO:
        2. selezione degli iperparametri dove richiesto tramite k-fold
        3. Scatter plot
        """








