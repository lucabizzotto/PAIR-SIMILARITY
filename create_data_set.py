import read_data
import random as rand
import numpy as np

def create_random_data(size):
    """
    create a random sample of documents
    :return:
    """
    corpus = read_data.get_data("corpus.npy")
    new_dict = {}
    for i in range(size):
        key, value = rand.choice(list(corpus.items()))
        new_dict[key] = value
    return new_dict


def data_augmentation(size):
    """
    increase size of document with similar documents so we can use later higher threshold
    :return:
    """
    # dictionay we are creating
    to_augment = {}
    to_augment = create_random_data(size)
    keys = to_augment.keys()
    l = [int(i) for i in keys]
    # max available index
    max_index = max(l)
    new_dict = {}
    # copy the last 0.2 documents
    for i in range(int(len(l)*0.8), len(l)):
        # new key
        max_index += 1
        to_augment[str(max_index)] = to_augment[str(l[i])]
    #np.save('/mnt/c/Users/bizzo/Desktop/AssigmentIII_LMD/Data/corpus_' + str(size) + '.npy', to_augment)














