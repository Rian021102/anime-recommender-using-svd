import copy
import numpy as np
def train_test_split(utility_data, test_size, random_state):
    full_data=copy.deepcopy(utility_data)
    np.random.seed(random_state)
    raw_ratings=full_data.raw_ratings
    np.random.shuffle(raw_ratings)

    threshold=int((1-test_size)*len(raw_ratings))

    train_raw_ratings=raw_ratings[:threshold]
    test_raw_ratings=raw_ratings[threshold:]

    full_data.raw_ratings=train_raw_ratings
    train_data=full_data.build_full_trainset()
    test_data=full_data.construct_testset(test_raw_ratings)

    return full_data, train_data, test_data