#!/usr/bin/env python
import numpy as np

def split_data(data, target, test_size = 0.33):

    # get the number of indices you want to assign to training
    n = int(len(data)*(1-test_size))

    # select indices for train/test at random
    indices = np.random.permutation(data.shape[0]) # rows
    idx_train, idx_test = indices[:n], indices[n:]

    # Set training and test sets
    X_train, X_test = data[idx_train], data[idx_test]
    y_train, y_test = target[idx_train], target[idx_test]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    # Testing
    seed = np.random.seed(0)        # set seed
    data = np.random.rand(10, 5)    # 10 x 5 matrix
    target = np.random.rand(10, 1)  # 10 x 1 matrix
    X_train, X_test, y_train, y_test = split_data(data, target)
    print(y_test)
     # [[ 0.57019677]
     # [ 0.46631077]
     # [ 0.16130952]
     # [ 0.2532916 ]]
