


import math
import numpy as np
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error


def main():
    # variables to the learning
    num_days = 15


    # TODO probably a bug in this section
    df = pd.read_csv("merged_data.csv", sep=',')
    df.drop('Date', axis=1, inplace=True)
    # set a new data frame with only close price
    data = df.filter(regex='(^Open.*$|^Close.*$|^High.*$|^Low.*$)')
    dataset = data.values



    # using 80% of the data to train round up
    training_data_len = math.ceil(len(dataset) * 0.8)
    #scaled_data1 = data.filter(regex='(^Open.*$|^Close.*$|^High.*$|^Low.*$)')
    scaled_data1 = data.filter(['Close_AAPL','Open_AAPL'])
    scaled_data2 = data.filter(items=['Close_AAPL'])
    #TODO check scaled data diffently
    scaled_data1.drop('Close_AAPL', axis=1, inplace=True)
    # scaling the data (helps the model)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    # computes the min and max values to scale and then scale the dataset based on those values
    scaled_data1 = scaled_data1.to_numpy()
    scaled_data2 = scaled_data2.to_numpy()
    # scaled_data1 = scaler.fit_transform(scaled_data1)
    # scaled_data2 = scaler2.fit_transform(scaled_data2)
    # create the training data set (scaled)
    train_data1 = scaled_data1[0:training_data_len, :]
    train_data2 = scaled_data2[0:training_data_len, :]
    train_data1 = scaler.fit_transform(train_data1)
    train_data2 = scaler2.fit_transform(train_data2)
    train_data = np.concatenate((train_data1, train_data2), axis=1)
    # split the data into x-train y_train
    x_train = []
    y_train = []
    # we are looking num_days days forward.
    # y contains data fom num_days and above, x from 0 to num_days
    for i in range(num_days, len(train_data)-7):
        x_train.append(train_data[i - num_days:i, :])
        y_train.append(train_data[i:i+7, :])
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data (LSTM model expects 3 dimensions so need to reshape to fit)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    # TODO: may need to smother the training data

    # model = load_model('my_model')
    # build the LSTM model
    model = Sequential()
    # adds a LSTM layer with ? neuruns
    ''' model.add(LSTM(1024, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=False))'''
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.05))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.05))
    model.add(Dense(50))
    model.add(Dense(1))

    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    # NOTE THIS TAKES TIME
    assert not np.any(np.isnan(x_train))
    model.fit(x_train, y_train, batch_size=1, epochs=20)
    model.save('my_model')
    # Creating the test data
    # TODO put elsewhere
    test_data1 = scaled_data1[training_data_len - num_days:, :]
    test_data2 = scaled_data2[training_data_len - num_days:, :]
    test_data1 = scaler.transform(test_data1)
    test_data2 = scaler2.transform(test_data2)
    test_data = np.concatenate((test_data1, test_data2), axis=1)

    # create x_test t_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(num_days, len(test_data)):
        x_test.append(test_data[i - num_days:i, :])
    # convert the data to numpy array
    x_test = np.array(x_test)
    # reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))
    # get the model predicted values
    predictions = model.predict(x_test)
    predictions = scaler2.inverse_transform(predictions)
    # get the RMSE (root mean squared error)
    y_test = y_test[:, 27]
    #mean = predictions.mean()
    total = 0
    print(predictions)
    for i in range(len(predictions)-1):
        if predictions[i] > predictions[i+1]:
            predictions[i] = 1
        else:
            predictions[i] = 0
        if predictions[i] == y_test[i]:
            total = total + 1
    accuracy = total / len(predictions)
    print("accuracy:")
    print(accuracy)
    # adds the result value after the model summary to a backlog file for follow up
    #backlog = open("backlog.txt", 'w+')
    #backlog.write(model.summary() + "\n" + "result= " + RMSE)
    #backlog.close()

    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close price')
    plt.plot(train['Close_AAPL'])
    plt.plot(valid[['Close_AAPL', 'Predictions']])
    plt.show()


if __name__ == '__main__':
    main()

