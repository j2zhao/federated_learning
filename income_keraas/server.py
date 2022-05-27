import flwr as fl
from typing import Dict, Tuple, List, Optional
import numpy as np
import time
from model import *



def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""


    X_test = np.load('testing_income/test_X.npy')
    y_test = np.load('testing_income/test_y.npy')

    def evaluate(parameters: fl.common.Weights):
        model.set_weights(parameters)
        true_loss, true_accuracy = model.evaluate(X_test, y_test) #TODO: Look into how to get model weight
        end = time.time()
        with open('log_accuracy.txt', 'a') as f:
            f.write("loss: {}, accuracy: {} \n".format(true_loss, true_accuracy))
        with open('log_performance.txt', 'a') as f:
            f.write("time: {} \n".format(end))
            
        return true_loss, {"accuracy": true_accuracy}

    return evaluate


if __name__ == "__main__":
    start = 0
    net = gen_net()
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=8,
        on_fit_config_fn=fit_round,
        eval_fn=get_eval_fn(net),
        min_fit_clients = 2
    )
    #fl.server.start_server("10.128.0.2:5040", strategy=strategy, config={"num_rounds": 300})
    #fl.server.start_server("localhost:8040", strategy=strategy, config={"num_rounds": 2})
    fl.server.start_server("10.128.0.2:5040", strategy=strategy, config={"num_rounds": 301})
    end = time.time()
    print(end-start)