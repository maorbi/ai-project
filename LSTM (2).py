# This is a sample Python script.

import math
import sys;

sys.path
import numpy as np
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD, Adam
import keras
import tensorboard
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


window_backward = 30

window_forward = 1


def main():
    df = pd.read_csv("merged_data.csv", sep=',')
    df.drop('Date', axis=1, inplace=True)
    # set a new data frame with only close price
    data = df.filter\
        (regex='(^Close.*$|^polarity.*$|^Volume.*$|^s&p_original.*$|^trend.*$|^trend_neg.*$)')
    print(data)
    dataset = data.values

    # using 80% of the data to train round up
    training_data_len = math.ceil(len(dataset) * 0.9)
    scaled_data1 = data.filter(regex='(^Close.*$|^Volume.*$)')
    scaled_data2 = data.filter(items=['trend', 'trend_neg'])
    # scaled_data1.drop('Close_^GSPC', axis=1, inplace=True)
    # scaling the data (helps the model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # computes the min and max values to scale and then scale the dataset based on those values
    scaled_data1 = scaled_data1.to_numpy()
    scaled_data2 = scaled_data2.to_numpy()
    scaled_data = np.concatenate((scaled_data2, scaled_data1), axis=1)
    # scaled_data1 = scaler.fit_transform(scaled_data1)
    # scaled_data2 = scaler2.fit_transform(scaled_data2)
    # create the training data set (scaled)
    train_data1 = scaled_data1[0:training_data_len + window_forward - 1, :]
    train_data2 = scaled_data2[0:training_data_len + window_forward - 1, :]
    train_data1 = scaler.fit_transform(train_data1)
    train_data1 = np.concatenate((train_data1, (data.filter(items=['polarity']).to_numpy())
    [0:training_data_len + window_forward - 1, :]), axis=1)
    # print(train_data2)
    # exit(0)
    train_data = np.concatenate((train_data2, train_data1), axis=1)
    print(train_data)
    x_train = []
    y_train = []
    # we are looking 60 days forward.
    # y contains data fom 60 and above, x from 0 to 60
    for i in range(window_backward, training_data_len):
        x_train.append(train_data[i - window_backward:i, :])
        y_train.append(train_data[i + window_forward - 1, :])
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data (LSTM model expects 3 dimensions so need to reshape to fit)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    # TODO: may need to smother the training data

    # build the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(2, activation='softmax'))
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    # train the model
    # NOTE THIS TAKES TIME
    assert not np.any(np.isnan(x_train))
    # y_train = np.asarray(train_data2).astype('float32').reshape((-1, 1))
    model.fit(x_train, y_train[:, [0, 1]], batch_size=5, epochs=55, shuffle=False)
    model.save('my_model')
    #model = load_model('my_model')
    # Creating the test data
    # TODO put elsewhere
    test_data1 = scaled_data1[training_data_len - window_backward:, :]
    test_data2 = scaled_data2[training_data_len - window_backward:, :]
    test_data1 = scaler.transform(test_data1)
    test_data1 = np.concatenate((test_data1, (data.filter(items=['polarity']).to_numpy())
    [training_data_len - window_backward:, :]), axis=1)
    #    test_data2 = scaler2.transform(test_data2)
    test_data = np.concatenate((test_data2, test_data1), axis=1)

    # create x_test t_test
    x_test = []
    y_test = df.filter(items=['torg'])
    y_test = np.array(y_test)
    y_test = y_test[training_data_len:, :]
    for i in range(window_backward, len(test_data) - window_forward + 1):
        x_test.append(test_data[i - window_backward:i, :])
    #    y_test.append(test_data[i + window_forward - 1, :])
    # for i in range(window_forward):
    #    y_test.append(test_data[len(test_data)-1, :])
    # convert the data to numpy array
    x_test = np.array(x_test)
    # reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))
    # y_test = np.array(y_test)
    # get the model predicted values
    predictions = model.predict(x_test)
    # get the RMSE (root mean squared error)
    # y_test = y_test[:, [0, 1]]
    total = 0
    final = []
    predictions = np.array(predictions)
    np.savetxt("prediction.csv", predictions, delimiter=",")
    for i in range(len(predictions)):
        if predictions[i, 0] > 0.5:
            final.append(1)
        else:
            if predictions[i, 0] < 0.5:
                final.append(0)
            else:
                final.append(0.5)
        if final[i] == y_test[i+window_forward-1]:
            total = total + 1

    accuracy = total / len(predictions)
    print(final)
    print(y_test)
    print(accuracy)
    # RMSE = np.sqrt(mean_squared_error(predictions, y_test))
    # print(RMSE)

    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:len(dataset) - window_forward + 1]
    valid['Predictions'] = final
    plt.figure(figsize=(16, 8))
    plt.title('LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    #plt.plot(train['s&p_original'])
    plt.plot(valid['s&p_original'])
    for i in range(len(final)):
        if final[i] == 1:
            plt.axvline(x=i+training_data_len+window_forward-1, color='b')
        if final[i] == 0:
            plt.axvline(x=i+training_data_len+window_forward-1, color='r')
    plt.show()


if __name__ == '__main__':
    main()
