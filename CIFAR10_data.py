import numpy as np
import pickle

def load_CIFAR_batch(batch_dir):
    with open(batch_dir, 'rb') as batch_file:
        datadict = pickle.load(batch_file, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float')
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(root_dir):
    xs = []
    ys = []

    for batch_index in range(1,6):
        batch_dir = root_dir + '/data_batch_' + str(batch_index)
        X, Y = load_CIFAR_batch(batch_dir)
        xs.append(X)
        ys.append(Y)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(root_dir + '/test_batch')
    return Xtr, Ytr, Xte, Yte

def load_CIFAR10_sample(root_dir, num_train=49000, num_val=1000, num_test=10000, mean_subtr=True, norm=True):
    Xtr, Ytr, Xte, Yte = load_CIFAR10('datasets/cifar-10-batches-py')

    X_train = Xtr[range(num_train)]
    X_val = Xtr[range(num_train, num_train + num_val)]
    X_test = Xte[range(num_test)]

    y_train = Ytr[range(num_train)]
    y_val = Ytr[range(num_train, num_train + num_val)]
    y_test = Yte[range(num_test)]

    X_train = np.reshape(X_train, newshape=(num_train, -1))
    X_val = np.reshape(X_val, newshape=(num_val, -1))
    X_test = np.reshape(X_test, newshape=(num_test, -1))

    if mean_subtr:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    if norm:
        std = np.std(X_train, axis=0)
        X_train /= std
        X_val /= std
        X_test /= std

    return X_train, y_train, X_val, y_val, X_test, y_test
