import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DataAnalysis import DataAnalysis as dan
from Regression import Regression as reg

if __name__ == "__main__":
    dataset = pd.read_csv('Dataset/insurance.csv')
    plot = dan()

    # print(dataset.head())             # primi 5 elementi
    # print(dataset.info())             # info tipi del dataframe
    # print(len(dataset.index))         # n righe
    # print(len(dataset.columns))       n colonne

    # various plot
    # plot.dataPlot(dataset)

    # codifico tutte le colonne che sono oggetti anzichè numeri
    le = LabelEncoder()
    le.fit(dataset.sex.drop_duplicates())
    dataset.sex = le.transform(dataset.sex)
    # smoker or not
    le.fit(dataset.smoker.drop_duplicates())
    dataset.smoker = le.transform(dataset.smoker)
    # region
    le.fit(dataset.region.drop_duplicates())
    dataset.region = le.transform(dataset.region)

    objReg = reg(dataset)

    # correlation matrix
    # plot.correlationMatrix(dataset)

    # regression
    objReg.linearRegression()

    # random forest
    objReg.randomForest()

    # support vector regression
    objReg.supportVectorRegression()
