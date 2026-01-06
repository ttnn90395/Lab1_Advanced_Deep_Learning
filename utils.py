import numpy as np
import torch


def create_train_dataset():
    n_train = 100000
    max_train_card = 10
    X_train_tensors = [] # Temporary list to hold tensors
    y_train_tensors = [] # Temporary list to hold tensors

    for i in range(n_train):
        card_count = np.random.randint(1, max_train_card + 1)
        card = np.random.randint(1, 11, size=card_count)
        label = np.sum(card)
        padded_array = np.pad(card, (0, 10 - card_count), mode='constant', constant_values=0)
        X_train_tensors.append(torch.tensor(padded_array, dtype=torch.long))
        y_train_tensors.append(torch.tensor(label, dtype=torch.float32))
        #print(padded_array, label)

    return torch.stack(X_train_tensors), torch.stack(y_train_tensors)


def create_test_dataset():

    n_test_classes = 20
    n_test_per_class = 10000
    X_test_tensors = [] 
    y_test_tensors = [] 
    for i in range(n_test_classes):
        card_count = 5*(i+1)
        for j in range(n_test_per_class):
            card = np.random.randint(1, 11, size=card_count)
            label = np.sum(card)
            X_test_tensors.append(torch.tensor(card, dtype=torch.long)) 
            y_test_tensors.append(torch.tensor(label, dtype=torch.float32))
            #print(card, label)

    return X_test_tensors, y_test_tensors
