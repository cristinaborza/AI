# incluziunile clasice
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import reader
from scipy import ndimage
from sklearn import preprocessing
from sklearn.utils import shuffle

# citirea datelor
def DataSet(description_file, path, contains_labels = True):
    # pun in array-ul images imaginile, iar in label etichetele corespunzatoare 
    images = []
    labels = []
    if contains_labels:
        # cazul pentru train si validation
        with open(description_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                # pentru fiecare rand, citesc imaginea cu numele respectiv si o atasez la vecotrul de imagini
                img = ndimage.imread(path + row[0])
                images.append(img)
                labels.append(row[1])
        return np.array(images), np.array(labels)
    else:
        # cazul pentru test
        with open(description_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                img = ndimage.imread(path + row[0])
                images.append(img)
        return np.array(images)

# citesc datele
validation_data, validation_labels = DataSet('Proiect/validation.txt', 'Proiect/validation/')
train_data, train_labels = DataSet('Proiect/train.txt', 'Proiect/train/')
test_data = DataSet('Proiect/test.txt', 'Proiect/test/', False)

# normalizarea datelor
def normalizare(train_data, validation_data, test_data):
    # folosesc normalizarea standard
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)

    scaled_train_data = scaler.transform(train_data)
    scaled_validation_date = scaler.transform(validation_data)
    scaled_test_data = scaler.transform(test_data) 
    
    return scaled_train_data, scaled_validation_date, scaled_test_data

# normzalizez datele
train_data, validation_data, test_data = normalizare(np.reshape(train_data, (30001, 1024)), np.reshape(validation_data, (5000, 1024)), np.reshape(test_data, (5000, 1024)))
# dau reshape la array-uri pentru a avea dimensiunile bune
train_data = np.reshape(train_data, (30001, 32, 32, 1))
validation_data = np.reshape(validation_data, (5000, 32, 32, 1))
test_data = np.reshape(test_data, (5000, 32, 32, 1))

# construiesc modelul
model = Sequential([
    Conv2D(64, 3, activation="relu", padding= "same", input_shape=[32, 32, 1]),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2),
    Dropout(0.25),
    Conv2D(128, 3, activation="relu", padding="same"),
    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(2),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.6),
    Dense(9, activation="softmax")
]) 

# compilez modelul
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# antrenez modelul
history = model.fit(train_data, 
                    train_labels, 
                    epochs = 25,
                    validation_data = (validation_data, validation_labels))

def nameOfItem(idx):
    return "0" + str(35001 + idx) + ".png"

# pun in fisierul predictiile
def GenerateOutput(prediction):
    with open("submission.txt", "w") as out_file:
        out_file.write("id,label\n")
        for idx in range(len(test_data)):
            # pentru fiecare imagine de test ii atribui label-ul cel mai probabil
            label = np.argmax(prediction[idx])
            out_file.write(nameOfItem(idx) + "," + str(label.item()) + "\n")

# construiesc matricea de confuzie
def confusion_matrix(y_true, y_pred): 
    conf_matrix = np.zeros((9, 9)) 
    for i in range(len(y_true)): 
        conf_matrix[int(y_true[i]), int(y_pred[i])] += 1
    return conf_matrix

# afisez acuratetea pe datele de vaidare
print(model.evaluate(validation_data, validation_labels))
# fac predictiile si generez output-ul
prediction = model.predict(test_data)
GenerateOutput(prediction)

# afisez si matricea de confuzie
aux = model.predict(validation_data)
validation_pred = np.zeros((len(validation_data)))
for idx in range(len(validation_data)):
    validation_pred[idx] = np.argmax(aux[idx])
print(confusion_matrix(validation_labels, validation_pred))

# afisez un plot cu evolutia acuratetii si a functiei de loss
pd.DataFrame(history.history).plot() 
plt.grid(True) 
plt.show()
