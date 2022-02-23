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

    X_test = np.load('testing/test_X.npy')
    y_test = np.load('testing/test_y.npy')
    def evaluate(parameters: fl.common.Weights):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
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
        temp = time.time()
        with open(log, 'a+') as f:
            f.write(str(temp - end))
        return aggregated_weights


if __name__ == "__main__":
    log = "log.txt"
    start = time.time()
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = SaveModelStrategy(
        min_available_clients=8,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        min_fit_clients = 2
    )
    fl.server.start_server("localhost:8040", strategy=strategy, config={"num_rounds": 200})
    end = time.time()
    print(end-start)