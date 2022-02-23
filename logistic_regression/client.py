import warnings
import flwr as fl
import numpy as np
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils_2 as utils

train_dir_name = "training_income"
test_dir_name = "testing_income"
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Training finished for round {config['rnd']}")
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    num = sys.argv[1]
    Xtrain_dir = os.path.join(train_dir_name, "train" + num + "_X.npy")
    ytrain_dir = os.path.join(train_dir_name, "train" + num + "_y.npy")
    
    Xtest_dir = os.path.join(test_dir_name, "test_X.npy")
    ytest_dir = os.path.join(test_dir_name, "test_y.npy")

    X_train = np.load(Xtrain_dir)
    y_train = np.load(ytrain_dir)

    X_test = np.load(Xtest_dir)
    y_test = np.load(ytest_dir)

    model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
    )

    utils.set_initial_params(model)
    fl.client.start_numpy_client("10.128.0.2:5040", client=MnistClient())
