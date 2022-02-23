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

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
        }
        return config

    return fit_config

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    testdata = load_test('testing_mnist/test.pt')
    valdata = load_test('training_mnist/train4.pt')

    def evaluate(parameters: fl.common.Weights):
        set_model_parameters(model, parameters)
        loss, accuracy = test(model, valdata)
        true_loss, true_accuracy = test(model, testdata)
        with open('log_accuracy.txt', 'a') as f:
            f.write("loss: {}, accuracy: {} \n".format(true_loss, true_accuracy))
        return loss, {"accuracy": accuracy}

    return evaluate

class TimeStrategy(fl.server.strategy.FedAvg):

    def configure_fit(self, rnd, weights, client_manager):
        if start == 0:
            start = time.time()
        return super.configure_fit(rnd,weights, client_manager)


    def evaluate(self, weights):
        result = super().evaluate(weights)
        with open('log_performance.txt', 'a') as f:
            f.write("time: {} \n".format(end - start))
        return result



if __name__ == "__main__":
    start = 0
    net = Net().to(DEVICE)
    strategy = TimeStrategy(
        min_available_clients=4,
        on_fit_config_fn=fit_round,
        eval_fn=get_eval_fn(net),
        min_fit_clients = 2
    )
    fl.server.start_server("10.128.0.2:5040", strategy=strategy, config={"num_rounds": 100})
    end = time.time()
    print(end-start)