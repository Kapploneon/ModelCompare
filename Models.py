from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    MLPClassifier:
        {
            'hidden_layer_sizes': [(10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
                                   (100),
                                   (1),
                                   (30, 30, 30, 30),
                                   (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'max_iter': [200, 1000],
            'warm_start': [True, False]
        },

    LogisticRegression:
        {
            'C': [10 ** i for i in range(-1, 2)],
            'fit_intercept': [True, False],
            'penalty': ['l2', 'l1'],
            'random_state': [0],
            'solver': ['liblinear']
        },

    GaussianNB:
        {
            'priors': [None]
        },

    DecisionTreeClassifier:
        {
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 40],
            'min_samples_split': [2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },

    BaggingClassifier:
        {
            'n_estimators': [10, 15, 20],
            'bootstrap': [True, False],
            'warm_start': [True, False]
        },

    GradientBoostingClassifier:
        {
            'loss': ['deviance', 'exponential'],
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 4]
        },

    AdaBoostClassifier:
        {
            'n_estimators': [50, 100, 10],
            'algorithm': ['SAMME', 'SAMME.R'],

        },

    RandomForestClassifier:
        {
            'n_estimators': [10, 20, 40],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 50],
            'bootstrap': [True, False],
            'warm_start': [True, False],

        },

    # SVC:
    #     {
    #         'C': [10 ** i for i in range(-1, 2)],
    #         'kernel': ['rbf', 'linear', 'poly', 'sigmoid', ],
    #         'shrinking': [True, False],
    #         'probability': [True]
    #     },
    # LinearSVC:
    #     {
    #         'C': [10 ** i for i in range(-1, 2)],
    #         'dual': [True, False],
    #         'max_iter': [100, 200, 300]
    #     },

    KNeighborsClassifier:
        {
            'n_neighbors': [5, 10, 15],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30, 50, 80]
        }
}

names = {
    MLPClassifier: 'MLPClassifier',
    LogisticRegression: 'LogisticRegression',
    GaussianNB: 'GaussianNB',
    DecisionTreeClassifier: 'DecisionTreeClassifier',
    BaggingClassifier: 'BaggingClassifier',
    GradientBoostingClassifier: 'GradientBoostingClassifier',
    AdaBoostClassifier: 'AdaBoostClassifier',
    RandomForestClassifier: 'RandomForestClassifier',
    SVC: 'SVC',
    LinearSVC: 'LinearSVC',
    KNeighborsClassifier: 'KNeighborsClassifier'
}
