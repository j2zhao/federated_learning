import flwr as fl
from typing import Dict, Tuple, List, Optional
import numpy as np
import time
from model import *
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test(test_dir):
    testdata = torch.load(test_dir)
    testloader = DataLoader(testdata, batch_size=32)
    return testloader

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    testdata = load_test('testing/test.pt')
    valdata = load_test('training/train4.pt')

    def evaluate(parameters: fl.common.Weights):
        set_model_parameters(model, parameters)
        loss, accuracy = test(model, valdata)
        true_loss, true_accuracy = test(model, testdata)
        end = time.time()
        with open('log_accuracy.txt', 'a') as f:
            f.write("loss: {}, accuracy: {} \n".format(true_loss, true_accuracy))
        with open('log_performance.txt', 'a') as f:
            f.write("time: {} \n".format(end))
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    start = 0
    net = Net().to(DEVICE)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=8,
        on_fit_config_fn=fit_round,
        eval_fn=get_eval_fn(net),
        min_fit_clients = 8
    )
    fl.server.start_server("localhost:8040", strategy=strategy, config={"num_rounds": 301})
    #fl.server.start_server("10.128.0.2:5040", strategy=strategy, config={"num_rounds": 300})
    end = time.time()
    print(end-start)