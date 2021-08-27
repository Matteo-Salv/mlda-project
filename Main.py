import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DataAnalysis import DataAnalysis as dan

if __name__ == "__main__":
    dataset = pd.read_csv('Dataset/insurance.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # print(dataset.head())             # primi 5 elementi
    # print(dataset.info())             # info tipi del dataframe
    # print(len(dataset.index))         # n righe
    # print(len(dataset.columns))       n colonne

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

    #scatterplot
    dan.scatterPlot(dataset)
    #various plot
    dan.dataPlot(dataset)