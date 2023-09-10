import numpy as np
from utility_matrix import get_utility_matrix

def split_train_test(rating_data, test_size=0.2, random_state=42):
    '''

    This function takes in the rating data and splits it into train and test data.

    '''




    np.random.seed(random_state)

    raw_index=rating_data.index.tolist().copy()

    np.random.shuffle(raw_index)

    threshold=int(len(raw_index)*(1-test_size))

    train_index=raw_index[:threshold]

    test_index=raw_index[threshold:]

    coo_data_train=get_utility_matrix(rating_data = rating_data.loc[train_index])
    coo_data_test=get_utility_matrix(rating_data = rating_data.loc[test_index])

    return coo_data_train, coo_data_test