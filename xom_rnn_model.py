from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def generate_exxon_rnn(start_dates, end_dates):
    # Gather data from Yahoo Finance
    xom = yf.Ticker('XOM')
    xom_data = xom.history(start=start_dates, end=end_dates)
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(start=start_dates, end=end_dates)
    cl = yf.Ticker('CL=F')
    cl_data = cl.history(start=start_dates, end=end_dates)

    # Combine the data of the three stocks
    data = pd.DataFrame({
        'XOM_Close': xom_data['Close'].values,
        'SP500_Close': sp500_data['Close'].values,
        'Oil_Close': cl_data['Close'].values,

        'XOM_Open': xom_data['Open'].values,
        'SP500_Open': sp500_data['Open'].values,
        'Oil_Open': cl_data['Open'].values,

        'XOM_High': xom_data['High'].values,
        'SP500_High': sp500_data['High'].values,
        'Oil_High': cl_data['High'].values,

        'XOM_Low': xom_data['Low'].values,
        'SP500_Low': sp500_data['Low'].values,
        'Oil_Low': cl_data['Low'].values,

        'XOM_Volume': xom_data['Close'].values,
        'SP500_Volume': sp500_data['Close'].values,
        'Oil_Volume': cl_data['Close'].values
        # Add other features if needed
    })

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset
    sequence_length = 60  # Length of input sequence -> 4 days 3 stocks 5 features each
    x = []
    y = []

    for i in range(len(data_scaled) - sequence_length - 5):  # 5 days in the future
        x.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length + 4, 0])

    x = np.array(x)
    y = np.array(y)

    # Build the RNN model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.simpleRNN(units=50, activation='tanh', return_sequences=True,
                                        input_shape=(x.shape[1], x.shape[2])))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=36, batch_size=32)

    # Test the model
    x_test = x[-1]
    y_test = y[-1]
    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
    y_pred = model.predict(x_test)
    print("The predicted price is: " + str(scaler.inverse_transform(y_pred)[0][0]))
    print("The actual price is: " + str(scaler.inverse_transform(y_test.reshape(-1, 1))[0][0]))
    print("The difference is: " + str(
        scaler.inverse_transform(y_pred)[0][0] - scaler.inverse_transform(y_test.reshape(-1, 1))[0][0]))

    # Save the model
    model.save('xom_rnn_model.h5')
    print("Model saved successfully")
