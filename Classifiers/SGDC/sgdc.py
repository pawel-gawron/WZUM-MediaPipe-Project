import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from lightgbm import LGBMClassifier
import missingno as msno
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn import svm, model_selection
import seaborn as sns
from sklearn.utils.metaestimators import _BaseComposition
import operator
import string
from sklearn import datasets
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import optuna
from skopt import BayesSearchCV

import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

def classifier(X_train, y_train, X_test, y_test):
    clfs = [
        LinearSVC,
        SVC,
        RandomForestClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        AdaBoostClassifier,
        BaggingClassifier,
        GaussianNB,
        BernoulliNB,
        LogisticRegression,
        SGDClassifier,
        NuSVC
    ]

    results = dict()
    print("len: ", len(X_train))
    print("len: ", len(y_train))

    for clf in clfs:
        mdl = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('classifier', clf())
        ])
        mdl.fit(X_train, y_train)
        print(clf.__name__)
        print(mdl.score(X_test, y_test))
        results[clf.__name__] = mdl.score(X_test, y_test)

        predict = mdl.predict(X_test)
        cm = confusion_matrix(y_test, predict)
        disp_cm = ConfusionMatrixDisplay(cm, display_labels=np.unique(mediapipe['letter']))
        # print(f'confusion_matrix: \n{confusion_matrix(y_test, predict)}')
        disp_cm.plot()
        disp_cm.ax_.set_title(clf.__name__)
        # plt.show()

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("GradientBoostingClassifier: ", score)

    clf = SVC(verbose=True)
    clf = LinearSVC(verbose=True)
    clf.fit(X_train, y_train)
    #preds = clf.predict(X_test)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

def letter_to_int(letter):
    letter = str(letter[0]).lower()
    alphabet = string.ascii_lowercase
    if letter in alphabet:
        return str(alphabet.index(letter) + 1)  # Dodajemy 1, aby uzyskać liczby od 1 do 26
    else:
        return None

def objective(trial, X_test, y_test):
 # Przykładowe parametry do optymalizacji
    penalty = trial.suggest_categorical('penalty', [None,'l2'])
    C = trial.suggest_float('C', 0.0001, 1000.0, log=True)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 100000)
    multi_class = trial.suggest_categorical('multi_class', ['auto', 'ovr'])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    
    # Tworzymy model z danymi parametrami
    model = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, solver=solver, 
                               max_iter=max_iter, multi_class=multi_class, class_weight=class_weight)
    
    # Wykonujemy ocenę wydajności na podstawie walidacji krzyżowej
    scores = cross_val_score(model, X_test, y_test, cv=5)
    
    # Zwracamy średnią dokładność (accuracy)
    return scores.mean()

if __name__ == '__main__':
    mediapipe = pd.read_excel('../WZUM dataset.xlsx', sheet_name="Main")
    # df = pd.concat(mediapipe)
    # print(mediapipe)

    for index, row in mediapipe.iterrows():
        mediapipe.at[index, 'letter'] = letter_to_int([mediapipe.at[index, 'letter']])

    # Wyodrębnienie nazw kolumn
    columns = mediapipe.columns.tolist()

    # Wybór kolumn do usunięcia
    # columnsErase = columns[127 : 129+1]
    columnsErase = columns[64 : 129+1]

    X = mediapipe.drop(columns=columnsErase)

    # # Columns to erase
    columns_to_remove = [col for col in X.columns if col.endswith('.z')]

    # # Erase columns
    X = X.drop(columns=columns_to_remove)
    X = X.drop(columns=mediapipe.columns[0],axis=1)
    # X.to_excel('saved_file.xlsx', index = False)
    y = mediapipe['letter'].astype(float)

##############################################################################################################
    mediapipe = pd.read_csv('/home/pawel/Documents/RISA/sem1/WZUM/WZUM_2023_DataGatherer/sample_dataset.csv')
    for index, row in mediapipe.iterrows():
        mediapipe.at[index, 'letter'] = letter_to_int([mediapipe.at[index, 'letter']])
    columns = mediapipe.columns.tolist()

    # Wybór kolumn do usunięcia
    # columnsErase = columns[127 : 129+1]
    columnsErase = columns[64 : 129+1]

    X_mlody = mediapipe.drop(columns=columnsErase)
    # Columns to erase
    columns_to_remove = [col for col in X_mlody.columns if col.endswith('.z')]

    # Erase columns
    X_mlody = X_mlody.drop(columns=columns_to_remove)
    X_mlody = X_mlody.drop(columns=mediapipe.columns[0],axis=1)
    # X = X.astype(float)
    y_mlody = mediapipe['letter'].astype(float)

    # print(X_mlody)
    # print(y_test)
##############################################################################################################

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y,
                                                random_state=42,
                                                test_size=0.5)
    
    # scaler = StandardScaler()
    # X_test = scaler.fit_transform(X_test)
    
    lsvc = LogisticRegression(C=0.07759862671508713, penalty= None, tol= 0.014179225272840684, fit_intercept= True,
                                 solver = 'sag', multi_class = 'auto', max_iter = 1976, class_weight = 'balanced')
    clf = make_pipeline(StandardScaler(),
                        lsvc)
    
    clf.fit(X_train, y_train)
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    score_mlody = clf.score(X_mlody, y_mlody)
    print("Score mlody: ", score_mlody)
    print("Score train: ", score_train)
    print("Score test: ", score_test)
    
##############################################################################################################

    # # Tworzymy obiekt Study z domyślnym algorytmem TPE
    # study = optuna.create_study(direction='maximize')

    # # Ustawienie wartości początkowych dla parametrów
    # study.enqueue_trial({'penalty': None})
    # study.enqueue_trial({'C': 33.384245216190436})
    # study.enqueue_trial({'fit_intercept': False})
    # study.enqueue_trial({'solver': 'sag'})
    # study.enqueue_trial({'max_iter': 83560})
    # study.enqueue_trial({'multi_class': 'auto'})
    # study.enqueue_trial({'class_weight': None})

    # # Uruchamiamy optymalizację
    # study.optimize(lambda trial: objective(trial, X_test, y_test), n_trials=100)

    # # Ocena wydajności na podstawie walidacji krzyżowej
    # best_model = LogisticRegression(**study.best_params)
    # scores = cross_val_score(best_model, X_test, y_test, cv=5)
    # print("Średnia dokładność (accuracy): ", scores.mean())

    # # Wyświetlamy najlepsze znalezione parametry
    # print("Najlepsze parametry: ", study.best_params)

    # # classifier(X_train, y_train, X_test, y_test)