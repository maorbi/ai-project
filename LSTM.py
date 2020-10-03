# This is a sample Python script.

import math
import sys; sys.path
import numpy as np
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error

def main():
    df = pd.read_csv("merged_data.csv", sep=',')
    df.drop('Date', axis=1,inplace=True)
    # set a new data frame with only close price
    data = df.filter(regex='$')
    dataset = data.values
    # using 80% of the data to train round up
    training_data_len = math.ceil(len(dataset) * 0.8)
    scaled_data2 = df.filter(items=['Close_A'])
    df.drop('Close_A', axis=1, inplace=True)
    scaled_data1 = data.filter(regex='$')
    # scaling the data (helps the model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    # computes the min and max values to scale and then scale the dataset based on those values
    scaled_data1 = scaler.fit_transform(scaled_data1)
    scaled_data2 = scaler2.fit_transform(scaled_data2)
    # create the training data set (scaled)
    train_data1 = scaled_data1[0:training_data_len, :]

    train_data2 = scaled_data2[0:training_data_len, :]
    train_data = np.concatenate((train_data1, train_data2),axis=1)
    # split the data into x-train y_train TODO: why x and y? what does it mean>?
    x_train = []
    y_train = []
    # we are looking 60 days forward.
    # y contains data fom 60 and above, x from 0 to 60
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, :])
        y_train.append(train_data[i, :])
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data (LSTM model expects 3 dimensions so need to reshape to fit)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    # TODO: may need to smother the training data

    # build the LSTM model
    model = Sequential()
    # adds a LSTM layer with 50 neuruns
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    # NOTE THIS TAKES TIME
    print(y_train)
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    # Creating the test data
    # TODO put elsewhere
    test_data1 = scaled_data1[training_data_len - 60:, :]
    test_data2 = scaled_data2[training_data_len - 60:, :]
    test_data = np.concatenate((test_data1, test_data2), axis=1)

    # create x_test t_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, :])
    # convert the data to numpy array
    x_test = np.array(x_test)
    # reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))
    # get the model predicted values
    print(x_test)
    predictions = model.predict(x_test)
    predictions = scaler2.inverse_transform(predictions)

    # get the RMSE (root mean squared error)
    y_test = y_test[:, 3]
    RMSE = np.sqrt(mean_squared_error(predictions, y_test))
    print(RMSE)

    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close price')
    plt.plot(train['Close_A'])
    plt.plot(valid[['Close_A', 'Predictions']])
    plt.show()


if __name__ == '__main__':
    main()

