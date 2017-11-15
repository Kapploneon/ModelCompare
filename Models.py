from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    LogisticRegression:
        {
            'C': [10 ** i for i in range(-1, 2)],
            'fit_intercept': [True, False],
            'penalty': ['l2', 'l1'],
            'random_state': [0],
            'solver': ['liblinear']
        },

    DecisionTreeClassifier:
        {
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 40],
            'min_samples_split': [2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }
}
