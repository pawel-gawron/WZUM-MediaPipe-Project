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
import pickle

class voteClassifier(_BaseComposition, BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='not probability', weights=None):
        self.classifiers = classifiers
        self.namedClassifiers = {clf.__class__.__name__: clf for clf in classifiers}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict(self, X):
        if self.vote == 'probability':
            maj_vote= np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(lambda x:
                                            np.argmax(np.bincount(x, weights=self.weights)),
                                            axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])

        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        if not deep:
            return super(voteClassifier,self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            return out

def letter_to_int(letter):
    letter = str(letter[0]).lower()
    alphabet = 'abcdefghiklmnopqrstuvwxy'
    if letter in alphabet:
        return str(alphabet.index(letter) + 1)  # Dodajemy 1, aby uzyskać liczby od 1 do 26
    else:
        return None

def classifier(X_train, y_train, X_test, y_test):
    clfs = [
        LinearSVC,
        SVC,
        RandomForestClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        LGBMClassifier,
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

def votingClassifierOwn(X_train, y_train, X_test, y_test):
    clf1 = LinearSVC()
    clf1 = CalibratedClassifierCV(clf1) 
    clf2 = RandomForestClassifier(n_estimators = 1000)
    clf3 = BaggingClassifier()
    clf4 = LGBMClassifier()
    clf5 = LogisticRegression()
    clf6 = NuSVC()

    pipe1 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf1]])
    
    pipe3 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf3]])
    
    pipe4 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf4]])
    
    pipe5 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf5]])
    
    pipe6 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf6]])
    
    clf_labels = ['LinearSVC', 'RandomForestClassifier', 'BaggingClassifier', 'LGBMClassifier', 'LogisticRegression', 'NuSVC']

    mv_clf = voteClassifier(classifiers=[pipe1, clf2, pipe3, pipe4, pipe5, pipe6])
    clf_labels += ["Glosowanie wiekszosciowe"]
    all_clf = [pipe1, clf2, pipe3, pipe4, pipe5, pipe6, mv_clf]

    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy', error_score='raise')
        print("Obszar pod krzywą ROC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        # clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

def votingClassifierSklearn(X_train, y_train, X_test, y_test):
    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    clf4 = LGBMClassifier()
    clf5 = LogisticRegression()
    clf6 = NuSVC()
    pipe1 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf1]])
    
    pipe3 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf3]])
    
    pipe4 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf4]])
    
    pipe5 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf5]])
    
    pipe6 = Pipeline([['sc', StandardScaler()],
                    ['min_max_scaler', MinMaxScaler()],
                    ['clf', clf6]])
    eclf1 = VotingClassifier(estimators=[
            ('lSVC', pipe1), ('RFC', clf2), ('BC', pipe3),
            ('LGBMC', pipe4), ('LR', pipe5), ('NuSVC', pipe6)],
            voting='hard')
    eclf1 = eclf1.fit(X_train, y_train)
    print(eclf1.score(X_test, y_test))

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)
    # print(study.best_trial)


def objective(trial, X_test, y_test):
    # Przykładowe parametry do optymalizacji
    C = trial.suggest_int('C', 1, 2000)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    degree = trial.suggest_int('degree', 1, 7)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) or trial.suggest_float('gamma', 0.001, 2.0)
    coef0 = trial.suggest_float('coef0', -2.0, 2.0)
    shrinking = trial.suggest_categorical('shrinking', [True, False])
    probability = trial.suggest_categorical('probability', [True, False])
    tol = trial.suggest_float('tol', 1e-5, 1.2)
    cache_size = trial.suggest_int('cache_size', 1, 1000)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    verbose = trial.suggest_categorical('verbose', [True, False])
    decision_function_shape = trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
    max_iter = -1  
    
    # Tworzymy model z danymi parametrami
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                verbose=verbose, decision_function_shape=decision_function_shape, max_iter=max_iter)
    
    # Wykonujemy ocenę wydajności na podstawie walidacji krzyżowej
    scores = cross_val_score(model, X_test, y_test, cv=5)
    
    # Zwracamy średnią dokładność (accuracy)
    return scores.mean()


if __name__ == '__main__':
    mediapipe = pd.read_excel('WZUM dataset.xlsx', sheet_name="Main")
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

    # X_train = X.head(4897) 
    # y_train = y.head(4897)

    # # print(y_train)

    # X_test = X.tail(240) 
    # y_test = y.tail(240)
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
                                                    test_size=0.1)
    
    # scaler = StandardScaler()
    # X_test = scaler.fit_transform(X_test)

    # print(X_test)
    
    # print("Rozmiar y_test: ", len(y_test))
    # print("Rozmiar X_test: ", len(X_test))
    
    # print(type(y_train))
    
    # msno.matrix(X_train)
    # plt.show()

    # classifier(X_train, y_train, X_test, y_test)
    # votingClassifierSklearn(X_train, y_train, X_test, y_test)
    # votingClassifierOwn(X_train, y_train, X_test, y_test)


    lsvc = SVC(C = 1492, cache_size = 323, class_weight = 'balanced', coef0 = 0.030309819659264665,
                    decision_function_shape = 'ovo', degree = 2, gamma = 'scale', kernel = 'poly',
                    max_iter = -1, probability = False, random_state = None, shrinking = False,
                    tol = 0.5031226880993588, verbose = False)
    clf = make_pipeline(StandardScaler(),
                        lsvc)
    
    clf.fit(X_train, y_train)
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    score_mlody = clf.score(X_mlody, y_mlody)
    print("Score mlody: ", score_mlody)
    print("Score train: ", score_train)
    print("Score test: ", score_test)

    with open('best_clf.pkl', 'wb') as file:
        pickle.dump(clf, file) ## clf = load do wczytywania z pliku modelu



    # # Definiujemy klasyfikator, dla którego będziemy dobierać parametry
    # classifier = SVC()

    # # Definiujemy zakresy poszczególnych parametrów do optymalizacji
    # param_space = {'C': (0.1, 1000.0),
    #             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #             'degree': (1, 5),
    #             'gamma': (0.01, 1.0, 'log-uniform'),
    #             'coef0': (0.0, 1.0),
    #             'shrinking': [True, False],
    #             'probability': [True, False],
    #             'tol': (1e-5, 1e-3, 'log-uniform'),
    #             'class_weight': [None, 'balanced']}

    # # Wykonujemy optymalizację Bayesowską
    # opt = BayesSearchCV(classifier, param_space, n_iter=1000, cv=5)
    # opt.fit(X_test, y_test)

    # # Wyświetlamy najlepsze znalezione parametry
    # print("Najlepsze parametry: ", opt.best_params_)

    # # Ocena wydajności na podstawie walidacji krzyżowej
    # scores = cross_val_score(opt.best_estimator_, X, y, cv=5)
    # print("Średnia dokładność (accuracy): ", scores.mean())



############################################################################################################################
    # # Tworzymy obiekt Study z domyślnym algorytmem TPE
    # study = optuna.create_study(direction='maximize')

    # # Ustawienie wartości początkowych dla parametrów
    # study.enqueue_trial({'C': 1492})
    # study.enqueue_trial({'gamma': 'scale'})
    # study.enqueue_trial({'cache_size': 323})
    # study.enqueue_trial({'class_weight': 'balanced'})
    # study.enqueue_trial({'coef0': 0.030309819659264665})
    # study.enqueue_trial({'decision_function_shape': 'ovo'})
    # study.enqueue_trial({'degree': 2})
    # study.enqueue_trial({'kernel': 'poly'})
    # study.enqueue_trial({'probability': False})
    # study.enqueue_trial({'shrinking': False})
    # study.enqueue_trial({'tol': 0.5031226880993588})
    # study.enqueue_trial({'verbose': False})

    # # Uruchamiamy optymalizację
    # study.optimize(lambda trial: objective(trial, X, y), n_trials=1000)

    # # Ocena wydajności na podstawie walidacji krzyżowej
    # best_model = SVC(**study.best_params)
    # scores = cross_val_score(best_model, X_test, y_test, cv=5)
    # print("Średnia dokładność (accuracy): ", scores.mean())

    # # Wyświetlamy najlepsze znalezione parametry
    # print("Najlepsze parametry: ", study.best_params)
