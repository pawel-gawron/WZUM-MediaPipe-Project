import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import pandas as pd
from sklearn.pipeline import Pipeline
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

import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

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
    alphabet = string.ascii_lowercase
    if letter in alphabet:
        return str(alphabet.index(letter) + 1)  # Dodajemy 1, aby uzyskać liczby od 1 do 26
    else:
        return None
    
# 1. Define an objective function to be maximized.
def objective(trial):
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
    # columns_to_remove = [col for col in X.columns if col.endswith('.z')]

    # # Erase columns
    # X = X.drop(columns=columns_to_remove)
    X = X.drop(columns=mediapipe.columns[0],axis=1)
    # X.to_excel('saved_file.xlsx', index = False)
    y = mediapipe['letter']

    # 2. Suggest values for the hyperparameters using a trial object.
    classifier_name = trial.suggest_categorical('classifier', ['LinearSVC'])
    if classifier_name == 'LinearSVC':
         svc_c = trial.suggest_float('svc_c', 1e-1, 5, log=True)
         classifier_obj = LinearSVC(C=svc_c)
    # else:
    #     rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    #     classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)
    
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


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
    # columns_to_remove = [col for col in X.columns if col.endswith('.z')]

    # # Erase columns
    # X = X.drop(columns=columns_to_remove)
    X = X.drop(columns=mediapipe.columns[0],axis=1)
    # X.to_excel('saved_file.xlsx', index = False)
    y = mediapipe['letter']

    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=0,
                                                    test_size=0.2)
    
    # print(type(y_train))
    
    # msno.matrix(X_train)
    # plt.show()
    
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
    # print("len: ", len(X_train))
    # print("len: ", len(y_train))

    # for clf in clfs:
    #     mdl = Pipeline([
    #         ('standard_scaler', StandardScaler()),
    #         ('classifier', clf())
    #     ])
    #     mdl.fit(X_train, y_train)
    #     print(clf.__name__)
    #     print(mdl.score(X_test, y_test))
    #     results[clf.__name__] = mdl.score(X_test, y_test)

    # # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    # #     max_depth=1, random_state=0).fit(X_train, y_train)
    # # score = clf.score(X_test, y_test)
    # # print("GradientBoostingClassifier: ", score)

    # # clf = SVC(verbose=True)
    # # clf = LinearSVC(verbose=True)
    # # clf.fit(X_train, y_train)
    # # #preds = clf.predict(X_test)
    # # train_score = clf.score(X_train, y_train)
    # # test_score = clf.score(X_test, y_test)

    #     predict = mdl.predict(X_test)
    #     cm = confusion_matrix(y_test, predict)
    #     disp_cm = ConfusionMatrixDisplay(cm, display_labels=np.unique(mediapipe['letter']))
    #     # print(f'confusion_matrix: \n{confusion_matrix(y_test, predict)}')
    #     disp_cm.plot()
    #     disp_cm.ax_.set_title(clf.__name__)
    #     # plt.show()

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
    
    # clf_labels = ['LinearSVC', 'RandomForestClassifier', 'BaggingClassifier', 'LGBMClassifier', 'LogisticRegression', 'NuSVC']

    # mv_clf = voteClassifier(classifiers=[pipe1, clf2, pipe3, pipe4, pipe5, pipe6])
    # clf_labels += ["Glosowanie wiekszosciowe"]
    # all_clf = [pipe1, clf2, pipe3, pipe4, pipe5, pipe6, mv_clf]

    # for clf, label in zip(all_clf, clf_labels):
    #     scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy', error_score='raise')
    #     print("Obszar pod krzywą ROC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    #     # clf.fit(X_train, y_train)
    #     print(clf.score(X_test, y_test))


    # clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    # clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    # clf3 = GaussianNB()
    # eclf1 = VotingClassifier(estimators=[
    #         ('lSVC', pipe1), ('RFC', clf2), ('BC', pipe3),
    #         ('LGBMC', pipe4), ('LR', pipe5), ('NuSVC', pipe6)],
    #         voting='hard')
    # eclf1 = eclf1.fit(X_train, y_train)
    # print(eclf1.score(X_test, y_test))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)


