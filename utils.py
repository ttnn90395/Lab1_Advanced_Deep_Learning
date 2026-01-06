import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10
    X_train = []
    y_train = []

    for i in range(n_train):
        card_count = np.random.randint(1, max_train_card + 1)
        card = np.random.randint(1, 11, size=card_count)
        label = np.sum(card)
        padded_array = np.pad(card, (0, 10 - card_count), mode='constant', constant_values=0)
        X_train.append(padded_array)
        y_train.append(label)
        print(padded_array, label)
    
    return X_train, y_train


def create_test_dataset():
    
    n_test_classes = 20
    n_test_per_class = 10000
    X_test = []
    y_test = []
    for i in range(n_test_classes):
        card_count = np.random.randint(1, 5*(i+1) + 1)
        for j in range(n_test_per_class):
            card = np.random.randint(1, 11, size=card_count)
            label = np.sum(card)
            X_test.append(card)
            y_test.append(label)
            #print(card, label)

    return X_test, y_test
