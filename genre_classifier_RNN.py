import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

DATA_PATH1 = "Data(40)1.json"
DATA_PATH2 = "Data(40)2.json"
DATA_PATH3 = "Data(40)3.json"
DATA_PATH4 = "Data(40)4.json"
DATA_PATH5 = "BollyData(20).json"

def load_data(data_path1,data_path2,data_path3,data_path4,data_path5):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(data_path1, "r") as fp:
        data = json.load(fp)

    X1 = np.array(data["mfcc"])
    y1 = np.array(data["labels"])

    with open(data_path2, "r") as fp:
        data = json.load(fp)

    X2 = np.array(data["mfcc"])
    y2 = np.array(data["labels"])

    with open(data_path3, "r") as fp:
        data = json.load(fp)

    X3 = np.array(data["mfcc"])
    y3 = np.array(data["labels"])

    with open(data_path4, "r") as fp:
        data = json.load(fp)

    X4 = np.array(data["mfcc"])
    y4 = np.array(data["labels"])

    with open(data_path5, "r") as fp:
        data = json.load(fp)

    X5 = np.array(data["mfcc"])
    y5 = np.array(data["labels"])

    Xn1 = np.vstack((X1, X2))
    Xn2 = np.vstack((Xn1,X3))
    Xn3 = np.vstack((Xn2, X4))
    X = np.vstack((Xn3, X5))
    yn1 = np.concatenate((y1, y2))
    yn2 = np.concatenate((yn1, y3))
    yn3 = np.concatenate((yn2, y4))
    y = np.concatenate((yn3,y5))
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH1, DATA_PATH2, DATA_PATH3, DATA_PATH4, DATA_PATH5)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates RNN-LSTM model

    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(128))

    # dense layer
    model.add(keras.layers.Dense(168, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(148, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(15, activation='softmax'))

    return model

def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    # X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    X = X.reshape(1, X.shape[0], X.shape[1])
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

if __name__ == "__main__":

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13

    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30,callbacks=[reduce_lr])
    model.save('music_genre_model2.h5')
    # plot accuracy/error for training and validation
    plot_history(history)

    X_to_predict = X_test[1]
    y_to_predict = y_test[1]
    print("Input Sample shape",X_to_predict)

    # predict sample
    predict(model, X_to_predict, y_to_predict)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

