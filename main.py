import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class CeditCardFraudDetection:

    def __init__(self):
        self.data = pd.read_csv('./datasets/creditcard.csv')

    def data_preparation(self):
        # print(data.shape)
        # print(data.describe())

        data = self.data

        fraud = data[data['Class'] == 1]
        valid = data[data['Class'] == 0]
        outlierFraction = len(fraud) / float(len(valid))
        print(outlierFraction)
        print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
        print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

        print('###########################')

        print('Amount details of the fraudulent transaction')
        print(fraud.Amount.describe())

        print()
        print('###########################')
        print()

        print('details of valid transaction')
        print(valid.Amount.describe())

        # Correlation matrix
        corrmat = data.corr()
        fig = plt.figure(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)
        plt.show()

    def regression_model(self):
        data = self.data

        # Aufteilen der Daten in Features (X) und Zielvariable (y)
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Aufteilen der Daten in Trainings- und Testsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialisierung und Training des Modells
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)

        # Vorhersagen auf dem Testdatensatz
        y_pred = model.predict(X_test)

        # Auswertung der Leistung
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))


fraud_detection = CeditCardFraudDetection()
fraud_detection.regression_model()
