import numpy as np


def get_data(file_name):
    # Load
    corpus = np.load('Data/' + file_name, allow_pickle='TRUE').item()
    return corpus
