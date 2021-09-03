from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model  import Ridge
import numpy as np
import matplotlib.pyplot as plt



class Regression:

    def __init__(self, dataset):
        x = dataset.drop(['charges'], axis=1)
        y = dataset.charges
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=0)
        scalerX = preprocessing.MinMaxScaler()
        self.x_train = scalerX.fit_transform(self.x_train)
        self.x_test = scalerX.transform(self.x_test)

    def linearRegression(self):
        lr = LinearRegression().fit(self.x_train, self.y_train)
        y_test_pred = lr.predict(self.x_test)
        print("### RESULTS FOR LINEAR REGRESSION ###")
        print('Accuracy: ' + str(lr.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))  # np = numpy
        print("Model MAE: ", err)
        plt.title('Linear Regression')
        plt.scatter(self.y_test, y_test_pred)
        tmp = [min(np.concatenate((self.y_test, y_test_pred))),
               max(np.concatenate((self.y_test, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()



    def randomForest(self):
        rfr = RandomForestRegressor(n_estimators=100, max_features='auto')
        rfr.fit(self.x_train, self.y_train)
        y_test_pred = rfr.predict(self.x_test)
        print("### RESULTS FOR RANDOM FOREST REGRESSION ###")
        print('Accuracy: ' + str(rfr.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        plt.figure(figsize=(8, 6))
        plt.title('Random Forest')
        plt.scatter(self.y_test, y_test_pred)
        tmp = [min(np.concatenate((self.y_test, y_test_pred))),
               max(np.concatenate((self.y_test, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()


    def supportVectorRegression(self):

        grid = {'C': np.logspace(-4, 3, 10),'kernel': ['rbf'],'gamma': np.logspace(-2, 2, 10), 'epsilon': [0, 0.1]}

        CV = GridSearchCV(estimator=SVR(),param_grid=grid,scoring='neg_mean_absolute_error',cv=10, verbose=0)

        H = CV.fit(self.x_train, self.y_train)

        svr = SVR(C = H.best_params_['C'],kernel='rbf' , gamma= H.best_params_['gamma'], epsilon=H.best_params_['epsilon'])

        svr.fit(self.x_train, self.y_train)
        y_test_pred = svr.predict(self.x_test)
        print("### RESULTS FOR SUPPORT VECTOR REGRESSION ###")
        print('Selected hyperparameters: ')
        print('C = %.3f, gamma = %.3f' % ((H.best_params_['C'])), ((H.best_params_['gamma'])))
        print('Accuracy: ' + str(svr.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        plt.title('SVR')
        plt.scatter(self.y_test, y_test_pred)
        tmp = [min(np.concatenate((self.y_test, y_test_pred))),
               max(np.concatenate((self.y_test, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()




    def KRLS(self):

        grid = {'alpha': np.logspace(-4, 3, 10), 'kernel': ['rbf'], 'gamma': np.logspace(-4, 3, 10)}

        CV = GridSearchCV(estimator=KernelRidge(), param_grid=grid, scoring='neg_mean_absolute_error', cv=10, verbose=0)

        H = CV.fit(self.x_train, self.y_train)

        ridge = KernelRidge(alpha = H.best_params_['alpha'], kernel='rbf', gamma = H.best_params_['gamma'])

        ridge.fit(self.x_train, self.y_train)
        y_test_pred = ridge.predict(self.x_test)
        print("### RESULTS FOR KRLS ###")
        print('Selected hyperparameters: ')
        print('alpha = %.3f, gamma = %.3f' % ((H.best_params_['alpha'])), ((H.best_params_['gamma'])))
        print('Accuracy: ' + str(ridge.score(self.x_test, self.y_test)))
        err = np.mean(np.abs(self.y_test - y_test_pred))
        print("Model MAE: ", err)
        plt.title('KRLS')
        plt.scatter(self.y_test, y_test_pred)
        tmp = [min(np.concatenate((self.y_test, y_test_pred))),
               max(np.concatenate((self.y_test, y_test_pred)))]
        plt.plot(tmp, tmp, 'r')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()

        """
        TODO:
        1. possibilità di regressione solo per determinate categorie (fumatori non fumatori, obesi...)
        """






