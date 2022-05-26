from collections import OrderedDict

from model import *
import flwr as fl
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def load_data(train_dir, trainlb_dir, test_dir, testlb_dir, val_dir, vallb_dir):
    """Load CIFAR-10 (training and test set)."""
    test = np.load(test_dir)
    train = np.load(train_dir)
    test_lb = np.load(testlb_dir)
    train_lb = np.load(trainlb_dir)

    val = np.load(val_dir)
    val_lb = np.load(vallb_dir)

    train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.)

    train_gen = train_datagen.flow(train, train_lb, batch_size=64)
    val_gen = test_datagen.flow(val, val_lb, batch_size=64)
    test_gen = test_datagen.flow(test, test_lb, batch_size=64)

    return train_gen, test_gen, val_gen




class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return net.get_weights()

    def fit(self, parameters, config):
        net.set_weights(parameters)
        #steps = 10
        #net.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]

        net.fit(trainloader, epochs=1, validation_data=valloader, callbacks=[callbacks])
        return net.get_weights(), 3810, {} # secound argument len(train)

    def evaluate(self, parameters, config):
        net.set_weights(parameters)
        loss, accuracy = net.evaluate(testloader)
        return loss, 1000, {"accuracy": accuracy} # secound argument len(test)


if __name__ == "__main__":

# Load model and data
    net = gen_net()
    test_dir = 'testing/test.npy'
    testlb_dir = "testing/testlb.npy"
    val_dir = 'training/train8.npy'
    vallb_dir = 'training/trainlb8.npy'

    train_dir = 'training/train' + sys.argv[1] + '.npy'
    trainlb_dir = 'training/trainlb' + sys.argv[1] + '.npy'

    trainloader, testloader, valloader = load_data(train_dir, trainlb_dir, test_dir, testlb_dir, val_dir, vallb_dir)
    fl.client.start_numpy_client("10.128.0.2:5040", client=CifarClient())
    #fl.client.start_numpy_client("localhost:8040", client=CifarClient())