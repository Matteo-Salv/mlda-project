import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis:

    def correlationMatrix(self, dataset):
        # stampo il coefficiente di correlazione rispetto al costo
        print(dataset.corr()['charges'].sort_values())
        # mostro l'heatmap
        sns.heatmap(data=dataset.corr(), cmap='coolwarm')
        plt.show()

    def dataPlot(self, dataset):
        # dalla correlazione notiamo un costo più alto per i fumatori. Ne stampiamo la distribuzione del costo
        nSmokers = dataset['smoker'].value_counts()  # indice 0 non fumatori, 1 fumatori
        f = plt.figure(figsize=(12, 5))
        # smokers
        ax = f.add_subplot(121)
        sns.histplot(dataset[(dataset.smoker == "yes")]["charges"], color='c', ax=ax)
        ax.set_title('Distribution of charges over a total of ' + str(nSmokers[1]) + ' smokers')
        # non smokers
        ax = f.add_subplot(122)
        sns.histplot(dataset[(dataset.smoker == "no")]['charges'], color='b', ax=ax)
        ax.set_title('Distribution of charges over a total of ' + str(nSmokers[0]) + ' non-smokers')
        plt.show()

        # numero di fumatori e non fumatori tra uomini e donne
        sns.catplot(x="smoker", kind="count", hue='sex', palette="pink", data=dataset, order=["no", "yes"])
        plt.show()

        # distribuzione d'età
        plt.figure(figsize=(12, 5))
        plt.title("Distribution of age over a total of " + str(len(dataset.index)) + " people")
        ax = sns.histplot(dataset["age"], color='g')
        plt.show()

        # numero di fumatori e non fumatori tra 18enni
        sns.catplot(x="smoker", kind="count", hue='sex', palette="rainbow", data=dataset[(dataset.age == 18)])
        plt.title("The number of smokers and non-smokers (18 years old)")
        plt.show()

        # numero di fumatori e non fumatori tra 64enni
        sns.catplot(x="smoker", kind="count", hue='sex', palette="rainbow", data=dataset[(dataset.age == 64)])
        plt.title("The number of smokers and non-smokers (64 years old)")
        plt.show()

        # fascia di spesa tra 18enni
        plt.figure(figsize=(12, 5))
        plt.title("Box plot for charges 18 years old smokers")
        sns.boxplot(y="smoker", x="charges", data=dataset[(dataset.age == 18)], orient="h", palette='rocket')
        plt.show()

        # fascia di spesa tra 64enni
        plt.figure(figsize=(12, 5))
        plt.title("Box plot for charges 64 years old smokers")
        sns.boxplot(y="smoker", x="charges", data=dataset[(dataset.age == 64)], orient="h", palette='pink')
        plt.show()

        # distribuzione BMI
        plt.figure(figsize=(12, 5))
        plt.title("Distribution of bmi over a total of " + str(len(dataset.index)) + " people [>=30 -> obesity]")
        ax = sns.histplot(dataset["bmi"], color='m')
        plt.show()

        # costo per BMI > 30
        plt.figure(figsize=(12, 5))
        plt.title("Distribution of charges for patients with BMI greater than 30")
        ax = sns.histplot(dataset[(dataset.bmi >= 30)]['charges'], color='b')
        plt.show()

        # costo per BMI < 30
        plt.figure(figsize=(12, 5))
        plt.title("Distribution of charges for patients with BMI less than 30")
        ax = sns.histplot(dataset[(dataset.bmi < 30)]['charges'], color='b')
        plt.show()

        sns.lmplot(x="age", y="charges", hue = "smoker", data=dataset, palette='inferno_r', height=7, legend = False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="center left", borderaxespad=0.)
        plt.title("Scatter plot smokers and non smokers")
        plt.show()

        sns.lmplot(x="bmi", y="charges", hue="smoker", data=dataset, palette='magma', height=8)
        plt.title("Scatter plot of charges and bmi")
        plt.show()