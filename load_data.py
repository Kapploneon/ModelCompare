import pandas as pd
from tqdm import tqdm
from logging import getLogger

TRAIN_DATA = './data/dota2Train.csv'
TEST_DATA = './data/dota2Test.csv'

logger = getLogger(__name__)


def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    logger.debug('exit')
    return df


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
