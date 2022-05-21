import pandas as pd
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import sys
import os
import pickle


def train_test_array(value, split = 8):
    train_df = pd.read_csv('./covidx-cxr2/train.txt', sep=" ", header=None)
    train_df.columns = ['patient id', 'filename', 'class', 'data source']
    train_df = train_df.drop(['patient id', 'data source'], axis=1)

    #train_df = train_df.sample(n=num_samples, random_state=0)
    train_df = train_df.sample(frac = 1, random_state = 42)
    train_df = np.array_split(train_df, split)[value]

    num_samples = train_df.shape[0]

    test_df = pd.read_csv('./covidx-cxr2/test.txt', sep=" ", header=None)
    test_df.columns = ['id', 'filename', 'class', 'data source']
    test_df = test_df.drop(['id', 'data source'], axis=1)

    #train_path = 'input/covidx-cxr2/train/'
    #test_path = 'input/covidx-cxr2/test/'


    print(f"Negative and positive values of train: {train_df['class'].value_counts()}")
    print(f"Negative and positive values of test: {test_df['class'].value_counts()}")

    # Now we create the train_data and train_label that will be used for ImageDataGenerator.flow
    train_data = list()
    train_label = list()

    test_data = list()
    test_label = list()

    for _, row in train_df.iterrows():
        file_path = "./covidx-cxr2/train/" + row["filename"]
        cur_image = Image.open(file_path).convert('RGB')
        image_resized = cur_image.resize((200, 200))
        img_data = np.array(image_resized)
        train_data.append(img_data)
        if row["class"] == "positive":
            train_label.append(1)
        else:
            train_label.append(0)

    for _, row in test_df.iterrows():
        file_path = "./covidx-cxr2/test/" + row["filename"]
        cur_image = Image.open(file_path).convert('RGB')
        image_resized = cur_image.resize((200, 200))
        img_data = np.array(image_resized)
        test_data.append(img_data)
        if row["class"] == "positive":
            test_label.append(1)
        else:
            test_label.append(0)

    train_data = np.asarray(train_data).reshape(int(num_samples), 200, 200, 3)
    print(train_data.shape)

    test_data = np.asarray(test_data).reshape(400, 200, 200, 3)
    print(test_data.shape)
    return train_data, train_label, test_data, test_label

def save_data(data, dir, file_name):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    # if isinstance(data, list):
    #     for i, d in enumerate(data):
    #         file = os.path.join(dir, file_name + str(i) + ".npy")
    #         np.save(file, d)
    #         # data = torch.load(file)
    #         # trainloader = DataLoader(data, batch_size=32, shuffle=True)
    #         # print("success?")
    # else:
    file = os.path.join(dir, file_name + ".npy")
    np.save(file, data)


if __name__ == "__main__":
    
    # trainset,trainlb, testset , testlb = train_test_array(value=int(sys.argv[1])) # change split value if needed

    # save_data(trainset, "training", 'train' + sys.argv[1])
    # save_data(trainlb, "training", 'trainlb' + sys.argv[1])
    # save_data(testset, "testing", 'test')
    # save_data(testlb, "testing", 'testlb' + sys.argv[1])
    for i in range(8):
        trainset, trainlb, testset , testlb = train_test_array(value=i) # change split value if needed
        save_data(trainset, "training", 'train' + str(i))
        save_data(trainlb, "training", 'trainlb' + str(i))
        if i == 0:
            save_data(testset, "testing", 'test')
            save_data(testlb, "testing", 'testlb')
# train_datagen = ImageDataGenerator(rescale=1./255.,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    vertical_flip=True)
# test_datagen = ImageDataGenerator(rescale=1.0/255.)

# train_gen = train_datagen.flow(train_data, train_label, batch_size=32)
# test_gen = test_datagen.flow(test_data, test_label, batch_size=32)