import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def generate_exxon_ann(start_dates, end_dates):
    xom = yf.Ticker('XOM')
    xom_data = xom.history(start=start_dates, end=end_dates)
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(start=start_dates, end=end_dates)
    cl = yf.Ticker('CL=F')
    cl_data = cl.history(start=start_dates, end=end_dates)

    # Assuming each stock's data has the same columns: Date, Close, and 88 other features

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
    sequence_length = 60  # Length of input sequence
    x = []
    y = []

    for i in range(len(data_scaled) - sequence_length - 5):
        x.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length + 4, 0])  # Use Stock1_Close as the output

    x = np.array(x)
    y = np.array(y)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    # Build the RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation='tanh', input_shape=(sequence_length, 3)),
        tf.keras.layers.Dense(1)  # Output layer with 1 unit to predict Stock1_Close
    ])
    model.compile(optimizer='adam', loss='mse')
    # Train the model
    model.fit(x_train, y_train, epochs=36, batch_size=32, validation_split=0.2)
    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print('Test Loss:', loss)
