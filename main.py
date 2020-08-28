# This is a sample Python script.

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("AAPL.csv")
    print(df.shape)
    plt.figure(figsize=(16, 8))
    plt.title("close price history")
    plt.plot(df['Close'])
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close price", fontsize=18)
    plt.show()

    # set a new data frame with only close price
    data = df.filter(['Close'])
    dataset = data.values
    # using 80% of the data to train round up
    training_data_len = math.ceil(len(dataset)*0.8)

    # scaling the data (helps the model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # computes the min and max values to scale and then scale the dataset based on those values
    scaled_data = scaler.fit_transform(dataset)
    print(scaled_data)

    # create the training data set (scaled)
    train_data = scaled_data[0:training_data_len, :]
    # split the data into x-train y_train TODO: why x and y? what does it mean>?
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data (LSTM model expects 3 dimensions so need to reshape to fit)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # TODO: may need to smother the training data

    # build the LSTM model
    model = Sequential()
    # adds a LSTM layer with 50 neuruns
    model.add(LSTM(50, return_sequences=True,))


if __name__ == '__main__':
    main()

