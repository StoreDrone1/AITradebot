import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
#cant double call the optimizer
#from tensorflow.keras.optimizers import Adam


def train_model():
    # Download historical stock prices from Yahoo Finance
    ticker = "KSCP"
    df = yf.download(ticker, start="2022-02-11")

    # Preprocess the data
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Split the data into training and testing sets
    train_data = df.iloc[:int(df.shape[0]*0.8), :]
    test_data = df.iloc[int(df.shape[0]*0.8):, :]

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(150, input_shape=(train_data.shape[1], 1)))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer="adam")

    # Reshape the data for input into the LSTM model
    X_train = np.array(train_data).reshape((train_data.shape[0], train_data.shape[1], 1))
    X_test = np.array(test_data).reshape((test_data.shape[0], test_data.shape[1], 1))

    # Train the LSTM model
    model.fit(X_train, train_data['Close'], epochs=32, batch_size=32, verbose=2)
    return model, X_test
