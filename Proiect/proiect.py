import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import reader
from scipy import ndimage
from sklearn import preprocessing
from sklearn.utils import shuffle

def DataSet(description_file, path, contains_labels = True):
    images = []
    labels = []
    if contains_labels:
        with open(description_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                img = ndimage.imread(path + row[0])
                images.append(img)
                labels.append(row[1])
        return np.array(images), np.array(labels)
    else:
        with open(description_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                img = ndimage.imread(path + row[0])
                images.append(img)
        return np.array(images)

X_valid, y_valid = DataSet('Proiect/validation.txt', 'Proiect/validation/')
X_train, y_train = DataSet('Proiect/train.txt', 'Proiect/train/')
X_test = DataSet('Proiect/test.txt', 'Proiect/test/', False)

def normalizare(train_data, validation_data, test_data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)

    scaled_train_data = scaler.transform(train_data)
    scaled_validation_date = scaler.transform(validation_data)
    scaled_test_data = scaler.transform(test_data) 
    
    return scaled_train_data, scaled_validation_date, scaled_test_data

X_train, X_valid, X_test = normalizare(np.reshape(X_train, (30001, 1024)), np.reshape(X_valid, (5000, 1024)), np.reshape(X_test, (5000, 1024)))
X_train = np.reshape(X_train, (30001, 32, 32, 1))
X_valid = np.reshape(X_valid, (5000, 32, 32, 1))
X_test = np.reshape(X_test, (5000, 32, 32, 1))

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=[32,32,1]),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train, 
                    y_train, 
                    epochs = 70,
                    validation_data = (X_valid, y_valid), 
                    shuffle=True)

def nameOfItem(idx):
    return "0" + str(35001 + idx) + ".png"

def GenerateOutput(prediction):
    with open("submission.txt", "w") as out_file:
        out_file.write("id,label\n")
        for idx in range(len(X_test)):
            label = np.argmax(prediction[idx])
            out_file.write(nameOfItem(idx) + "," + str(label.item()) + "\n")

print(model.evaluate(X_valid, y_valid))
prediction = model.predict(X_test)
GenerateOutput(prediction)