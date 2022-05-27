from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import time
import sys

def get_test(dir = 'testing'):
    X_dir = os.path.join(dir, "test_X.npy")
    y_dir = os.path.join(dir, "test_y.npy") 
    return (np.load(X_dir), np.load(y_dir))

def get_train(dir = 'training', num = 10):
    Xtrains = []
    ytrains = []
    for i in range(num):
        X_dir = os.path.join(dir, "train" + str(i) +"_X.npy")
        y_dir = os.path.join(dir, "train" + str(i) +"_y.npy")
        Xtrains.append(np.load(X_dir))
        ytrains.append(np.load(y_dir))
    
    X_train = np.concatenate(Xtrains)
    y_train = np.concatenate(ytrains)
    return (X_train, y_train)

def train_logistic(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def result(model, X_test, y_test):
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = model.score(X_test, y_test)
    return (loss, accuracy)

if __name__ == "__main__":
    # change directory between mnist and housing
    dir_train = sys.argv[0] #training_mnist or training_income
    dir_test = sys.argv[1] #testing_mnist or testing_income
    n = int(sys.argv[2]) # amount of data: 4, 8
    xtrain, ytrain = get_train(dir = dir_train, num=n)
    xtest, ytest = get_test(dir = dir_test)
    start = time.time()
    model = train_logistic(xtrain, ytrain)
    loss, accuracy = result(model, xtest, ytest)
    end = time.time()
    print(accuracy)
    print(end - start)
