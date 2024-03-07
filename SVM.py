import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow.keras as keras
import matplotlib.pyplot as plt



DATA_PATH1 = "Data(20)1.json"
DATA_PATH2 = "Data(20)2.json"
DATA_PATH3 = "BollyData(20).json"

def load_data(data_path1,data_path2,data_path3):
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

    Xn1 = np.vstack((X1, X2))
    X = np.vstack((Xn1,X3))
    yn1 = np.concatenate((y1, y2))
    y = np.concatenate((yn1,y3))
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
    X, y = load_data(DATA_PATH1, DATA_PATH2, DATA_PATH3)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


if __name__ == "__main__":

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    # Before applying the StandardScaler, flatten the MFCC features
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_validation_flat = X_validation.reshape(X_validation.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Standardize your features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_validation_scaled = scaler.transform(X_validation_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Create an SVM classifier with a linear kernel
    svm_classifier = SVC(kernel='rbf', C=10.0)
    # Train the SVM model
    svm_classifier.fit(X_train_scaled, y_train)

    # Validation
    y_validation_pred = svm_classifier.predict(X_validation_scaled)
    validation_accuracy = accuracy_score(y_validation, y_validation_pred)
    print(f'Validation accuracy: {validation_accuracy * 100:.2f}%')

    # Testing
    y_test_pred = svm_classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')





