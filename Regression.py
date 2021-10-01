from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


class Regression:

    def __init__(self, dataset):
        x = dataset.drop(['charges'], axis=1)
        y = dataset.charges
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=0)

        # data normalization
        scalerX = preprocessing.MinMaxScaler()
        self.x_train = scalerX.fit_transform(self.x_train)
        self.x_test = scalerX.transform(self.x_test)

    def scatterPlot(self, y_test_pred, title):
        plt.title(title)
        plt.scatter(self.y_test, y_test_pred)
        tmp = [min(np.concatenate((self.y_test, y_test_pred))),
               max(np.concatenate((self.y_test, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()

    def linearRegression(self):
        lr = LinearRegression().fit(self.x_train, self.y_train)
        y_test_pred = lr.predict(self.x_test)
        print("### RESULTS FOR LINEAR REGRESSION ###")
        print('Accuracy (R^2): ' + str(lr.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        self.scatterPlot(y_test_pred, 'Linear Regression')

    def randomForest(self):
        rfr = RandomForestRegressor(n_estimators=100, max_features='auto')
        rfr.fit(self.x_train, self.y_train)
        y_test_pred = rfr.predict(self.x_test)
        print("### RESULTS FOR RANDOM FOREST REGRESSION ###")
        print('Accuracy (R^2): ' + str(rfr.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        plt.figure(figsize=(8, 6))
        self.scatterPlot(y_test_pred, 'Random Forest')

    def supportVectorRegression(self):
        # parameters selection phase
        grid = {'C': np.logspace(-4, 3, 10), 'kernel': ['rbf'], 'gamma': np.logspace(-4, 3, 10), 'epsilon': [0, 0.1]}
        CV = GridSearchCV(estimator=SVR(), param_grid=grid, scoring='neg_mean_absolute_error', cv=10, verbose=0)
        H = CV.fit(self.x_train, self.y_train)

        svr = SVR(C=H.best_params_['C'], kernel='rbf', gamma=H.best_params_['gamma'], epsilon=H.best_params_['epsilon'])
        svr.fit(self.x_train, self.y_train)
        y_test_pred = svr.predict(self.x_test)
        print("### RESULTS FOR SUPPORT VECTOR REGRESSION ###")
        print('Selected hyperparameters:')
        print("C = " + str(H.best_params_['C']) + ", gamma = " + str(H.best_params_['gamma']))
        print('Accuracy (R^2) : ' + str(svr.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        self.scatterPlot(y_test_pred, 'SVR')

    def KRLS(self):
        # parameters selection phase
        grid = {'alpha': np.logspace(-4, 3, 10), 'kernel': ['rbf'], 'gamma': np.logspace(-4, 3, 10)}
        CV = GridSearchCV(estimator=KernelRidge(), param_grid=grid, scoring='neg_mean_absolute_error', cv=10, verbose=0)
        H = CV.fit(self.x_train, self.y_train)
        krls = KernelRidge(alpha = H.best_params_['alpha'], kernel='rbf', gamma = H.best_params_['gamma'])
        krls.fit(self.x_train, self.y_train)
        y_test_pred = krls.predict(self.x_test)
        print("### RESULTS FOR KRLS ###")
        print('Selected hyperparameters: ')
        print("alpha = " + str(H.best_params_['alpha']) + ", gamma = " + str(H.best_params_['gamma']))
        print('Accuracy (R^2): ' + str(krls.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        self.scatterPlot(y_test_pred, 'KRLS')








