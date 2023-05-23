import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from lightgbm import LGBMClassifier
import missingno as msno
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm, model_selection
import seaborn as sns
from sklearn.utils.metaestimators import _BaseComposition
import operator
import string
from sklearn import datasets

import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

class voteClassifier(_BaseComposition, BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
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
    y = mediapipe['letter']

    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=0,
                                                    test_size=0.2)
    
    # print(type(y_train))
    
    msno.matrix(X_train)
    plt.show()
    
    clfs = [
        LinearSVC,
        SVC,
        RandomForestClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        LGBMClassifier,
        AdaBoostClassifier,
        BaggingClassifier
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
    clf2 = RandomForestClassifier()
    clf3 = BaggingClassifier()

    pipe1 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf1]])
    
    pipe3 = Pipeline([['sc', StandardScaler()],
                    ['clf', clf3]])
    
    # clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1)
    # clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    # clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

    # pipe1 = Pipeline([['sc', StandardScaler()],
    #                   ['clf', clf1]])
    
    # pipe3 = Pipeline([['sc', StandardScaler()],
    #                 ['clf', clf3]])
    
    clf_labels = ['LinearSVC', 'RandomForestClassifier', 'BaggingClassifier']

    # for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    #     scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy', error_score='raise')

    #     print("Obszar pod krzywą ROC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    classifiers_dict = {
    'clf1': pipe1,
    'clf2': clf2,
    'clf3': pipe3
}

    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y,
                                                random_state=1,
                                                test_size=0.5)

    mv_clf = voteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ["Glosowanie wiekszosciowe"]
    all_clf = [pipe1, clf2, pipe3, mv_clf]

    for clf, label in zip(all_clf, clf_labels):
        # print(clf)
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy', error_score='raise')
        print("Obszar pod krzywą ROC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    # estimator = []
    # estimator.append(clf1)
    # estimator.append(clf2)
    # estimator.append(clf3)

    # # Voting Classifier with hard voting
    # vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
    # vot_hard.fit(X_train, y_train)
    # y_pred = vot_hard.predict(X_test)
    
    # # using accuracy_score metric to predict accuracy
    # score = accuracy_score(y_test, y_pred)
    # print("Hard Voting Score % d" % score)
    
    # # Voting Classifier with soft voting
    # vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
    # vot_soft.fit(X_train, y_train)
    # y_pred = vot_soft.predict(X_test)
    
    # # using accuracy_score
    # score = accuracy_score(y_test, y_pred)
    # print("Soft Voting Score % d" % score)


