#By: https://github.com/Day1Hour0/Har_Tensorflow

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def get_data(filename, type):
    """ Read data
    Parm : filename : the input file name
    type : train or test

    Return the data in format data, label, channel of the data
    """
    path_ = os.path.join(filename, type )
    path_signals = os.path.join(path_, "Inertial Signals")

    sub_file = "y_" + type + ".txt"
    label_path = os.path.join(path_, sub_file)

    labels = pd.read_csv(label_path, header=None)

    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(type) + 5

    list_of_channels = []
    data_array = np.zeros((len(labels), 128, n_channels))
    count = 0
    for filter_channel in channel_files:
        channel_name = filter_channel[:-posix]
        raw_data = pd.read_csv(os.path.join(path_signals, filter_channel), delim_whitespace=True, header=None)
        data_array[:, :, count] = raw_data.to_numpy()
        list_of_channels.append(channel_name)
        count += 1
    return data_array, labels[0].values, list_of_channels

def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	return y

def standardize(data):
	""" Standardize data """
	std_data = (data - np.mean(data, axis=0)[None,:,:]) / np.std(data, axis=0)[None,:,:]
	return std_data

def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['acc'])
    plt.plot(epoch_range, history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

  # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def model ():
    #get data C:/Users/Last Hope/PycharmProjects

    train_data, train_label, ch_train = get_data(filename="./UCI HAR Dataset\\", type="train")
    test_data, test_label,ch_test = get_data(filename="./UCI HAR Dataset\\", type="test")

    X_train = standardize(train_data)
    X_test = standardize(test_data)

    #train shape : (7352, 128, 9)
    #test shape : (2947, 128, 9)

    X_train = X_train.reshape(7352, 128, 9, 1)
    X_test = X_test.reshape(2947, 128, 9, 1)

    Y_train = one_hot(train_label, n_class = 6)
    Y_test = one_hot(test_label, n_class = 6)
    print(X_train[0].shape)

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=64,kernel_size=(2,2), activation='relu', input_shape=X_train[0].shape),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Conv2D(filters=64,kernel_size=(2,2), activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(units=64,activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Dense(units=6, activation='softmax')
                                        ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    history = model.fit(X_train, Y_train, epochs = 100, validation_data=(X_test, Y_test))
    plot_learningCurve(history, 100)

model()
#print(os.path.dirname(os.path.abspath(__file__)))