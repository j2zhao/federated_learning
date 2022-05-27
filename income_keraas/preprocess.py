import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess(data):
    # remove space
    data.columns = [cols.replace(' ', '') for cols in data.columns]
    data["education"] = [cols.replace(' ', '') for cols in data["education"]]
    data["marital-status"] = [cols.replace(' ', '') for cols in data["marital-status"]]
    data["relationship"] = [cols.replace(' ', '') for cols in data["relationship"]]
    data["race"] = [cols.replace(' ', '') for cols in data["race"]]
    data["gender"] = [cols.replace(' ', '') for cols in data["gender"]]

    #missing data
    data = data.replace('?', np.nan)
    data.dropna(inplace=True,axis=0)

    #categorical value
    cat_columns = ['workclass','education','marital-status', 'occupation', 'relationship', 'race', 'gender','native-country']
    df_dumy = pd.get_dummies(data, columns = cat_columns)
    return df_dumy

def duplicate(data, n):
    data = pd.concat([data]*n)
    data.reset_index()
    return data

def split_train_test(data, percentage = 0.1):
    index = int(data.shape[0]*percentage)
    test = data[-index:]
    train = data[:-index]
    return train, test

def split_categories(data, num = 10, size = None, shuffle = True):
    if shuffle:
        data = data.sample(frac = 1, random_state = 42)
    
    if size == None:
        return np.array_split(data, num)
    else:
        raise NotImplemented()
    
def XY_split(data, name, folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    X = data.drop("income_>50K",axis=1)
    
    scaler = StandardScaler()
    X = X.to_numpy(dtype=np.float32)
    X = scaler.fit_transform(X)
    
    y = data["income_>50K"]
    X_file = os.path.join(folder_name, name + "_X.npy")
    y_file = os.path.join(folder_name, name + "_y.npy")
    np.save(X_file, X)
    np.save(y_file, y)

if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    print(train.shape)
    train = preprocess(train)
    train, test = split_train_test(train)
    print(train.shape)
    train = duplicate(train, 10)
    #print(train.shape)
    trains = split_categories(train, num = 8)
    training_dir = 'training_income'
    testing_dir = 'testing_income'
    for i, train in enumerate(trains):
        XY_split(train, 'train' + str(i), training_dir)
    
    XY_split(test, 'test', testing_dir)
