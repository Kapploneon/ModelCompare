import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc

from load_data import load_train_data, load_test_data
from Models import models, names

logger = getLogger(__name__)

DIR = 'result_tmp/'


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g


if __name__ == '__main__':

    log_fmt = Formatter(
        # '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s]'
        '%(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()

    x_train = df.drop(df.columns[0], axis=1)
    y_train = df[df.columns[0]].values

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    best_params = dict()

    # Traverse model and parameter set.
    for model, all_params in tqdm(models.items()):
        logger.info('\n\tTraining Model: {}'.format(model))
        min_score = 100
        min_params = None
        for params in list(ParameterGrid(all_params)):
            logger.info('\n\t\tparams: {}'.format(params))

            list_gini_score = []
            list_logloss_score = []
            list_accuracy_score = []
            all_preds = np.zeros(shape=y_train.shape[0])
            for train_idx, valid_idx in cv.split(x_train, y_train):
                trn_x = x_train.iloc[train_idx, :]
                val_x = x_train.iloc[valid_idx, :]

                trn_y = y_train[train_idx]
                val_y = y_train[valid_idx]

                clf = model(**params)
                clf.fit(trn_x, trn_y)
                pred = clf.predict_proba(val_x)[:, 1]
                # print(pred)
                sc_logloss = log_loss(val_y, pred)
                sc_gini = - gini(val_y, pred)
                sc_accuracy = clf.score(val_x, val_y)

                all_preds[valid_idx] = pred

                list_logloss_score.append(sc_logloss)
                list_gini_score.append(sc_gini)
                list_accuracy_score.append(sc_accuracy)
                logger.debug(
                    '\t\tlogloss: {}, gini: {}, accuracy: {}'.format(sc_logloss,
                                                                     sc_gini,
                                                                     sc_accuracy))
                break

            with open(DIR + 'all_preds.pkl', 'wb') as f:
                pickle.dump(all_preds, f, -1)

            sc_logloss = np.mean(list_logloss_score)
            sc_gini = np.mean(list_gini_score)
            sc_accuracy = np.mean(list_accuracy_score)
            if min_score > sc_gini:
                min_score = sc_gini
                min_params = params
            logger.info(
                '\t\tlogloss: {}, gini: {}, accuracy: {}'.format(sc_logloss,
                                                                 sc_gini,
                                                                 sc_accuracy))
            logger.info(
                '\t\tcurrent min score: {}, params: {}'.format(min_score,
                                                               min_params))

        logger.info('\n\tminimum params: {}'.format(min_params))
        logger.info('\tminimum gini: {}'.format(min_score))

        best_params[model] = min_params

        # clf = model(**min_params)
        # clf.fit(x_train, y_train)
        # with open(DIR + 'model.pkl', 'wb') as f:
        #     pickle.dump(clf, f, -1)
        #
        # logger.info('\tTraining end.\n')
        # with open(DIR + 'model.pkl', 'rb') as f:
        #     clf = pickle.load(f)

    print(best_params)

    df = load_test_data()

    x_test = df.drop(df.columns[0], axis=1)
    y_test = df[df.columns[0]].values

    gini_score = dict()
    list_logloss_score = []
    accuracy_score = dict()

    for model, params in best_params.items():
        gini_score[model] = []
        accuracy_score[model] = []

    for train_idx, valid_idx in tqdm(cv.split(x_test, y_test)):
        trn_x = x_train.iloc[train_idx, :]
        val_x = x_train.iloc[valid_idx, :]

        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]

        for model, params in best_params.items():
            clf = model(**params)
            clf.fit(trn_x, trn_y)
            pred = clf.predict_proba(val_x)[:, 1]
            sc_gini = - gini(val_y, pred)
            sc_accuracy = clf.score(val_x, val_y)

            gini_score[model].append(sc_gini)
            accuracy_score[model].append(sc_accuracy)
            pass
    for model, params in best_params.items():
        gini_score[model] = np.mean(gini_score[model])
        accuracy_score[model] = np.mean((accuracy_score[model]))
        logger.info('{} & {} & {} & {} \\\\'.format(names[model],
                                                    params,
                                                    accuracy_score[model],
                                                    gini_score[model]))

    print(gini_score)
    print(accuracy_score)

    logger.info('end')
