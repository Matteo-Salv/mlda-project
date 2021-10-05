import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DataAnalysis import DataAnalysis as dan
from Regression import Regression as reg


def subsetSelection(dataset):
    while True:
        print("do you want to subsample the dataset in order to analyze only particular categories "
              "(i.e. only smokers or BMI > 30)? [y/n]")
        sel = input()
        if sel == 'y':
            print("in which categories are you interested? Smoker [s], non smoker [o], bmi > 30 [b], bmi <= 30 [i]")
            sel = input()
            if sel == 's':
                dataset = dataset[dataset["smoker"] == 1]
            elif sel == 'o':
                dataset = dataset[dataset["smoker"] == 0]
            elif sel == 'b':
                dataset = dataset[dataset["bmi"] > 30]
            elif sel == 'i':
                dataset = dataset[dataset["bmi"] <= 30]
            else:
                continue
            break
        elif sel == 'n':
            break
    return dataset


if __name__ == "__main__":
    dataset = pd.read_csv('Dataset/insurance.csv')
    plot = dan()

    # eliminazione delle colonne children e region, che non consideriamo
    dataset = dataset.drop(["children", "region"], axis=1)

    # print(dataset.head())             # primi 5 elementi
    # print(dataset.info())             # info tipi del dataframe
    # print(len(dataset.index))         # n righe
    # print(len(dataset.columns))       n colonne

    # various plot
    plot.dataPlot(dataset)

    # codifico tutte le colonne che sono oggetti anzichè numeri
    le = LabelEncoder()
    le.fit(dataset.sex.drop_duplicates())
    dataset.sex = le.transform(dataset.sex)
    # smoker or not
    le.fit(dataset.smoker.drop_duplicates())
    dataset.smoker = le.transform(dataset.smoker)

    # selezione del subset
    #dataset = subsetSelection(dataset)

    # print(dataset.head())  # primi 5 elementi

    objReg = reg(dataset)

    # correlation matrix
    plot.correlationMatrix(dataset)

    # regression
    objReg.linearRegression()

    # random forest
    objReg.randomForest()

    # support vector regression
    objReg.supportVectorRegression()

    # kernel ridge regression
    objReg.KRLS()
