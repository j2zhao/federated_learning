import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from typing import Dict, Tuple, List, Optional
import time

import utils_2 as utils

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}

def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    X_test = np.load('testing_income/test_X.npy')
    y_test = np.load('testing_income/test_y.npy')
    def evaluate(parameters: fl.common.Weights):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        end = time.time()
        with open('log_accuracy.txt', 'a') as f:
            f.write("loss: {}, accuracy: {} \n".format(loss, accuracy))
        with open('log_performance.txt', 'a') as f:
            f.write("time: {} \n".format(end))
        return loss, {"accuracy": accuracy}

    return evaluate

class TimeStrategy(fl.server.strategy.FedAvg):

    def configure_fit(self, *args):
        if start == 0:
            start = time.time()
        return super.configure_fit(*args)

    def evaluate(self, *args):
        result = super().evaluate(*args)
        with open('log_performance.txt', 'a') as f:
            f.write("time: {} \n".format(end - start))
        return result 


if __name__ == "__main__":
    start = 0
    end = 0
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=8,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        min_fit_clients = 2
    )
    #10.128.0.2:5040
    fl.server.start_server("10.128.0.2:5040", strategy=strategy, config={"num_rounds": 301})
    #fl.server.start_server("localhost:8040", strategy=strategy, config={"num_rounds": 301})
    end = time.time()
    print(end-start)
