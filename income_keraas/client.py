from collections import OrderedDict

from model import *
import flwr as fl
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import os



class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return net.get_weights()

    def fit(self, parameters, config):
        net.set_weights(parameters)
        #steps = 10
        #net.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)

        net.fit(X_train, y_train, epochs=1)
        return net.get_weights(), 3810, {"accuracy": 0} # secound argument len(train)

    def evaluate(self, parameters, config):
        net.set_weights(parameters)
        loss, accuracy = net.evaluate(X_test, y_test)
        return loss, 1000, {"accuracy": accuracy} # secound argument len(test)


if __name__ == "__main__":

# Load model and data
    net = gen_net()
    num = sys.argv[1]
    train_dir_name = "training_income"
    test_dir_name = "testing_income"
    Xtrain_dir = os.path.join(train_dir_name, "train" + num + "_X.npy")
    ytrain_dir = os.path.join(train_dir_name, "train" + num + "_y.npy")
    
    Xtest_dir = os.path.join(test_dir_name, "test_X.npy")
    ytest_dir = os.path.join(test_dir_name, "test_y.npy")

    X_train = np.load(Xtrain_dir)
    y_train = np.load(ytrain_dir)

    X_test = np.load(Xtest_dir)
    y_test = np.load(ytest_dir)

    fl.client.start_numpy_client("10.128.0.2:5040", client=CifarClient())
    #fl.client.start_numpy_client("localhost:8040", client=CifarClient())