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

def letter_to_int(letter):
    letter = str(letter[0]).lower()
    alphabet = string.ascii_lowercase
    if letter in alphabet:
        return str(alphabet.index(letter) + 1)  # Dodajemy 1, aby uzyskać liczby od 1 do 26
    else:
        return None

def objective(trial, X_test, y_test):
    # Przykładowe parametry do optymalizacji
    C = trial.suggest_int('C', 1, 1100)
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
    tol = trial.suggest_float('tol', 1e-5, 1)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    intercept_scaling = trial.suggest_float('intercept_scaling', 1.0, 10.0)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    verbose = trial.suggest_categorical('verbose', [True, False])
    
    # Tworzymy model z danymi parametrami
    model = LinearSVC(C=C, penalty=penalty, tol=tol, fit_intercept=fit_intercept,
                      intercept_scaling=intercept_scaling, class_weight=class_weight,
                      verbose=verbose, max_iter=10000, dual=False)
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y,
                                                random_state=0,
                                                test_size=0.2)
    
    # scaler = StandardScaler()
    # X_test = scaler.fit_transform(X_test)
    
##############################################################################################################
    lsvc = LinearSVC(C = 726, penalty = 'l2', tol = 3.9210147932038824e-05, fit_intercept = True,
                    intercept_scaling = 9.944643307738232, class_weight = 'balanced', verbose = True)
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

    # # Uruchamiamy optymalizację
    # study.optimize(lambda trial: objective(trial, X_test, y_test), n_trials=500)

    # # Ocena wydajności na podstawie walidacji krzyżowej
    # best_model = LinearSVC(**study.best_params)
    # scores = cross_val_score(best_model, X_test, y_test, cv=5)
    # print("Średnia dokładność (accuracy): ", scores.mean())

    # # Wyświetlamy najlepsze znalezione parametry
    # print("Najlepsze parametry: ", study.best_params)