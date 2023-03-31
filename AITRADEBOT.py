#important note...if you want to trade a different stock you have to change the ticker here and 
# in "TrainFinancialModel.py" also.  Otherwise the neural net will train against the ticker that is 
# in the trainfinancialmodel and confuse the learning process.

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import alpaca_trade_api as trade_api
from alpaca_trade_api.rest import TimeFrame
#from alpaca_trade_api import Alpaca - why the flying fuck alpaca said this, keep in case the API changes
#their requirements 
import time as t
import schedule
from TrainFinancialModel import train_model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

#ticker symbol for apple, CHANGE THIS!
symbol = "KSCP"
# Connect to the Alpaca API
# API KEY ID , then SECRET KEY
# input api key / secret code respectively here
api = trade_api.REST('AK0O14FMSF66V38NRIF1', 'u0Wqplhwce1EADTaYJbpYJ9xUCh8MfJX3XZhpwhK')

# download historical stock prices from Yahoo Finance
ticker = "KSCP"
df = yf.download(ticker, start="2022-02-11")

lr = 0.01
optimizer = Adam(learning_rate=lr)
# Preprocess the data
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#print(df.tail(5))
# Split the data into training and testing sets use 90% of the data for training and 50% for testing
train_data = df.iloc[:int(df.shape[0]*0.9), :]
test_data = df.iloc[int(df.shape[0]*0.5):, :]
#print("Test data = {} Train data = {}".format(test_data, train_data))
# Define the LSTM model, for now use 150 cells, but could try more...not really sure
# experiment with the number of LSTM layers (cells), if you go too high and see poor results 
# lower the LSTM value, or the first value in the LSTM models parameter
model = Sequential()
model.add(LSTM(150, input_shape=(train_data.shape[1], 1)))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics='accuracy')

# Reshape the data for input into the LSTM model
X_train = np.array(train_data).reshape((train_data.shape[0], train_data.shape[1], 1))

# Train the LSTM model
try:
    model.fit(X_train, train_data['Close'], epochs=64, batch_size=128, verbose=2)
except Exception as e:
    print("Unable to fit model - Error {}".format(e))
#hopefully the optimizer conflict is resolved...
model, X_test = train_model()
print("Finished training, pending trades...")

model.save("Trained_KSCP.h5")

if type(symbol) == list and len(symbol) == 1:
        symbol = symbol[0]
#hold off on this shit....for some reason it throws an error
'''def process_data(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)'''

def trade():
    print("Thinking about trading")
    model, X_test = train_model()
    
    #get our balance before buying
    account = api.get_account()
    balance = float(account.cash)    
    
    # get the predictions from the updated model
    try:
        predictions = model.predict(X_test)
    
        # Get actual values
        actual_values = test_data['Close'].values

        # Get predicted values
        predicted_values = predictions.flatten()

        # Plot actual vs predicted values
        plt.plot(actual_values, label='Actual')
        plt.plot(predicted_values, label='Predicted')

        # Add title and axis labels
        plt.title('Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')

        # Show legend
        plt.legend()

        # Show plot
        plt.show()
    except Exception as e:
        print("Unable to generate predictions - Error {}".format(e))

    #pred = predictions[0]
    #get stock price
    barset = api.get_bars(symbol, TimeFrame.Day, limit=1)
    #not sure why we're losing this here, but CHECK TO FUCKING SEE
    if barset:
        symbol_bars = barset[0]
        stock_price = symbol_bars.c
    else:
        last_trade = api.get_latest_trade(symbol).price
        stock_price = float(last_trade)
        #print("Last trade price = {}".format(stock_price))
    #shares = int(amount / stock_price)
    #calculate number of shares that can be bought with the amount
    # assume that we want to buy if the prediction is positive and sell if the prediction is negative
    # but within a tolerated range as the output value is funky....may need to change this for different
    # stock tickers
   
    if predictions[0][0] >= 0.07:
        # place a buy order
        if balance > stock_price:
            api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        else:
            print("Insufficient funds for buy")
    elif predictions[0][0] <= 0.07:
        # place a sell order
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
    # keep for now, but probably wont need this unless i implement second stage backpropagation or 
    # some kind of periodicity
    with open("Predictions_Today.txt", 'a') as preds:
        preds.writelines(str(predictions[0][0]))
    print("Predictions[0]: ", predictions[0][0])
    #clear the barset.  Hopefully this will eliminate the error
    #that the barset index is out of bounds
    barset.clear()    

# schedule the trade function to run at certain intervals every day, trying different time frames 
# recommened by a stock trader

#schedule.every().day.at("19:00").do(trade)
#schedule.every().day.at("09:00").do(trade)
#schedule.every().day.at("10:00").do(trade)
schedule.every().day.at("11:52").do(trade)
schedule.every().day.at("12:01").do(trade)
schedule.every().day.at("13:00").do(trade)
#schedule.every().day.at("14:00").do(trade)
#schedule.every().day.at("15:00").do(trade)
schedule.every().day.at("16:00").do(trade)

while True: 
    schedule.run_pending()
    t.sleep(1)
   

