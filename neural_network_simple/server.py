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
        with open('log.txt', 'a') as f:
            f.write("loss: {}, accuracy: {} \n".format(true_loss, true_accuracy))
        return loss, {"accuracy": accuracy}

    return evaluate

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

if __name__ == "__main__":
    start = time.time()
    net = Net().to(DEVICE)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=4,
        on_fit_config_fn=fit_round,
        eval_fn=get_eval_fn(net),
        min_fit_clients = 2
    )
    fl.server.start_server("localhost:8080", strategy=strategy, config={"num_rounds": 100})
    end = time.time()
    print(end-start)