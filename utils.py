
import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10
    X_train = []
    y_train = []

    for i in range(n_train):
        card_count = np.random.randint(1, max_train_card + 1)
        card = np.random.choice(range(1, 11), size=card_count, replace=False)
        label = np.sum(card)
        X_train.append(card)
        y_train.append(label)
    
    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################

    return X_test, y_test
