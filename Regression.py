from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import SVR


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

    def randomForest(self):
        rfr = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
        rfr.fit(self.x_train, self.y_train)
        y_train_pred = rfr.predict(self.x_train)
        y_test_pred = rfr.predict(self.x_test)
        print("### RESULTS FOR RANDOM FOREST REGRESSION ###")
        print('MSE train data: %.3f, MSE test data: %.3f' %
              (metrics.mean_squared_error(y_train_pred, self.y_train),
               metrics.mean_squared_error(y_test_pred, self.y_test)))

    def supportVectorRegression(self):
        svr = SVR(kernel='rbf')
        svr.fit(self.x_train, self.y_train)
        y_train_pred = svr.predict(self.x_train)
        y_test_pred = svr.predict(self.x_test)
        print("### RESULTS FOR SUPPORT VECTOR REGRESSION ###")
        print('MSE train data: %.3f, MSE test data: %.3f' %
              (metrics.mean_squared_error(y_train_pred, self.y_train),
               metrics.mean_squared_error(y_test_pred, self.y_test)))

    def KRLS(self):
        """
        TODO:
        1. funzione KRLS
        2. selezione degli iperparametri dove richiesto tramite k-fold
        3. Scatter plot
        """






