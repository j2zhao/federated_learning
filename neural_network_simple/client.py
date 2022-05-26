from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from model import *
import flwr as fl
import sys

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(train_dir, test_dir):
    """Load CIFAR-10 (training and test set)."""
    test = torch.load(test_dir)
    train = torch.load(train_dir)
    trainloader = DataLoader(train, batch_size=32, shuffle=True)
    testloader = DataLoader(test, batch_size=32)
    num_examples = {"trainset" : len(train), "testset" : len(test)}
    return trainloader, testloader, num_examples

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return get_model_parameters(net)

    # def set_parameters(self, parameters):
    #     params_dict = zip(net.state_dict().keys(), parameters)
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #     net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        set_model_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        set_model_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


if __name__ == "__main__":
# Load model and data
    

    net = Net().to(DEVICE)
    test_dir = 'testing/test.pt'
    train_dir = 'training/train' + sys.argv[1] + '.pt'
    trainloader, testloader, num_examples = load_data(train_dir, test_dir)
    fl.client.start_numpy_client("10.128.0.2:5040", client=CifarClient())
    #fl.client.start_numpy_client("localhost:8040", client=CifarClient())