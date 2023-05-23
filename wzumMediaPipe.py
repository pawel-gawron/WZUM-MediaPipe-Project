import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from lightgbm import LGBMClassifier
import missingno as msno
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import svm, model_selection
import seaborn as sns

import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == '__main__':
    mediapipe = pd.read_excel('WZUM dataset.xlsx', sheet_name="Main")
    # df = pd.concat(mediapipe)
    # print(mediapipe)

    # Wyodrębnienie nazw kolumn
    columns = mediapipe.columns.tolist()

    # Wybór kolumn do usunięcia
    # columnsErase = columns[127 : 129+1]
    columnsErase = columns[64 : 129+1]

    X = mediapipe.drop(columns=columnsErase)

    # # Columns to erase
    # columns_to_remove = [col for col in X.columns if col.endswith('.z')]

    # # Erase columns
    # X = X.drop(columns=columns_to_remove)
    X = X.drop(columns=mediapipe.columns[0],axis=1)
    y = mediapipe['letter']

    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=0,
                                                    test_size=0.2)
    
    msno.matrix(X_train)
    plt.show()
    
    clfs = [
        LinearSVC,
        SVC,
        RandomForestClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        LGBMClassifier
    ]

    results = dict()
    # print("len: ", len(X_train))
    # print("len: ", len(y_train))

    for clf in clfs:
        mdl = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('classifier', clf())
        ])
        mdl.fit(X_train, y_train)
        print(clf.__name__)
        print(mdl.score(X_test, y_test))
        results[clf.__name__] = mdl.score(X_test, y_test)

    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    #     max_depth=1, random_state=0).fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    # print("GradientBoostingClassifier: ", score)

    # clf = SVC(verbose=True)
    # clf = LinearSVC(verbose=True)
    # clf.fit(X_train, y_train)
    # #preds = clf.predict(X_test)
    # train_score = clf.score(X_train, y_train)
    # test_score = clf.score(X_test, y_test)

        predict = mdl.predict(X_test)
        cm = confusion_matrix(y_test, predict)
        disp_cm = ConfusionMatrixDisplay(cm, display_labels=np.unique(mediapipe['letter']))
        # print(f'confusion_matrix: \n{confusion_matrix(y_test, predict)}')
        disp_cm.plot()
        disp_cm.ax_.set_title(clf.__name__)
        # plt.show()

