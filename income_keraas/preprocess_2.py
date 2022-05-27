import numpy as np
import pandas as pd
import os
import openml


def preprocess():
    mnist_openml  = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    return Xy

def duplicate(data, n):
    data = pd.concat([data]*n)
    data.reset_index()
    return data

def split_train_test(data):
    train = data[:60000]
    test = data[60000:]
    return train, test

def split_categories(data, num = 10, size = None, shuffle = True):
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(data)
    
    if size == None:
        return np.array_split(data, num)
    else:
        raise NotImplemented()
    
def XY_split(data, name, folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    X = data[:, :-1]
    y = data[:, -1]
    # X = data.drop("income_>50K",axis=1)
    # y = data["income_>50K"]
    X_file = os.path.join(folder_name, name + "_X.npy")
    y_file = os.path.join(folder_name, name + "_y.npy")
    np.save(X_file, X)
    np.save(y_file, y)

if __name__ == "__main__":
    #train = pd.read_csv("train.csv")
    #print(train.shape)
    train = preprocess()
    train, test = split_train_test(train)
    print(train.shape)
    #print(train.shape)
    train = duplicate(train, 10)
    print(train.shape)
    trains = split_categories(train, num = 8)
    training_dir = 'training_mnist'
    testing_dir = 'testing_mnist'
    for i, train in enumerate(trains):
        XY_split(train, 'train' + str(i), training_dir)
    
    XY_split(test, 'test', testing_dir)