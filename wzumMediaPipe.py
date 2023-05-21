import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions

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
    y = mediapipe['letter']

    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=0,
                                                    test_size=0.2)
    
    clfs = [
        LinearSVC,
        SVC,
        RandomForestClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier
    ]

    results = dict()
    print("len: ", len(X_train))
    print("len: ", len(y_train))

    for clf in clfs:
        mdl = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('min_max_scaler', MinMaxScaler()),
            ('classifier', clf())
        ])
        mdl.fit(X_train, y_train)
        print(clf.__name__)
        print(mdl.score(X_test, y_test))
        results[clf.__name__] = mdl.score(X_test, y_test)

    plot_decision_regions(np.array(X_train), np.array(y_train),
                          clf=mdl, legend=1)
    plt.show()

    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap='viridis')
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()
