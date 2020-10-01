
import math
import numpy as np
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, concatenate
import matplotlib.pyplot as plt

'''
this function reads and organizes the data from the file into a numpy array
input: 
    datafile: the location of the file to read in csv format
    desired columns: a list of strings of the parameters other than "Close"
:returns a scaled numpy array, and scaler for "close"
'''
def read_normalize_data(datafile,desired_columns):
    df = pd.read_csv(datafile)
    assert "Close" not in desired_columns
    scaled_data1 = df.filter(items=desired_columns)
    scaled_close1 = df.filter(items=["Close"])
    # scaling the data (helps the model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler = MinMaxScaler(feature_range=(0,1))
    # computes the min and max values to scale and then scale the dataset based on those values
    scaled_data = scaler.fit_transform(scaled_data1)
    scaled_close = close_scaler.fit_transform(scaled_close1)
    scaled_data = np.concatenate((scaled_data,scaled_close),axis=1)
    return  scaled_data ,close_scaler

'''
this function builds both the test and train sets and prepares them through scaling the lists
input: 
    dataframe: a pandas array of samples and features
    num_of_days: how many days backwards the model will look
    feature_num: the number of features being transferred including "Close"
'''
def train_test_data_build(dataframe, num_of_days, feature_num=4 ):
    # using 80% of the data to train round up
    training_data_len = math.ceil(len(dataframe) * 0.8)
    train_data = dataframe[:training_data_len, :]
    # split the data into x-train y_train
    x_train = []
    y_train = []
    # we are looking num_of_days backwards.
    # y contains data fom i and above, x from 0 to num_of_days
    for i in range(num_of_days, len(train_data)):
        x_train.append(train_data[i - num_of_days:i, :])
        y_train.append(train_data[i, feature_num-1])
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # create x_test t_test
    x_test = []
    y_test = dataframe[training_data_len:, feature_num-1]
    test_data = dataframe[training_data_len: , :]
    for i in range(num_of_days, len(test_data)):
        x_test.append(test_data[i - num_of_days:i, :])
    # convert the data to numpy array
    x_test , y_test = np.array(x_test) ,np.array(y_test)
    # Creating the test data

    return x_train, y_train ,x_test , y_test

def build_LSTM_model( x_train , y_train, feature_num ,num_epochs=1):
    # build the LSTM model
    model = Sequential()
    # adds a LSTM layer with 50 neuruns
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], feature_num)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    # NOTE THIS TAKES TIME
    model.fit(x_train, y_train, batch_size=1, epochs=num_epochs)
    return model


def main():

    data , close_scaler = read_normalize_data("AAPL.csv",['Open','High','Low'])
    x_train ,y_train , x_test , y_test = train_test_data_build(data,30,4)
    print(x_test.shape)
    model = build_LSTM_model(x_train, y_train,4,1)
    # get the model predicted values
    predictions = model.predict(x_test)
    predictions = close_scaler.inverse_transform(predictions)

    # get the RMSE (root mean squared error)
    RMSE = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(RMSE)

'''
    training_len = len(x_train)
    # plot the data
    train = data[:training_len]
    valid = data[training_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close price')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.show()
'''

if __name__ == '__main__':
    main()

