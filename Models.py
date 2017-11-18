from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier

models = {
    MLPClassifier:
        {
            'hidden_layer_sizes': [(5, 2), (10, 4), (30, 5)],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'max_iter': [200, 1000]
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
            # 'n_estimators': [10, 15, 20],
            'bootstrap': [True, False],
            # 'warm_start': [True, False]
        },

    GradientBoostingClassifier:
        {
            'loss': ['deviance', 'exponential'],
            # 'n_estimators': [50, 100, 200],
            # 'max_depth' : [2,3,4]
        },

    AdaBoostClassifier:
        {
            'n_estimators': [50, 100, 10],
            'algorithm': ['SAMME', 'SAMME.R'],

        }
}

names = {
    MLPClassifier: 'MLPClassifier',
    LogisticRegression: 'LogisticRegression',
    GaussianNB: 'GaussianNB',
    DecisionTreeClassifier: 'DecisionTreeClassifier',
    BaggingClassifier: 'BaggingClassifier',
    GradientBoostingClassifier: 'GradientBoostingClassifier',
    AdaBoostClassifier: 'AdaBoostClassifier'
}
