import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# This file is for the RNN model for Chevron
def generate_chevron_rnn(start_dates, end_dates):
    num_epochs_to_decay = 10
    cvx = yf.Ticker('CVX')
    cvx_data = cvx.history(start=start_dates, end=end_dates)
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(start=start_dates, end=end_dates)
    cl = yf.Ticker('CL=F')
    cl_data = cl.history(start=start_dates, end=end_dates)

    # Combine the data of the three stocks
    data = pd.DataFrame({
        'CVX_Close': cvx_data['Close'].values,
        'SP500_Close': sp500_data['Close'].values,
        'Oil_Close': cl_data['Close'].values,

        'CVX_Open': cvx_data['Open'].values,
        'SP500_Open': sp500_data['Open'].values,
        'Oil_Open': cl_data['Open'].values,

        'CVX_High': cvx_data['High'].values,
        'SP500_High': sp500_data['High'].values,
        'Oil_High': cl_data['High'].values,

        'CVX_Low': cvx_data['Low'].values,
        'SP500_Low': sp500_data['Low'].values,
        'Oil_Low': cl_data['Low'].values,

        'CVX_Volume': cvx_data['Volume'].values,
        'SP500_Volume': sp500_data['Volume'].values,
        'Oil_Volume': cl_data['Volume'].values
        # Add other features if needed
    })

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset
    sequence_length = 4
    x, y = [], []

    for i in range(len(data_scaled) - sequence_length):
        x.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length][0])

    x, y = np.array(x), np.array(y)
    num_train_samples = int(0.9 * len(x))
    x_train, x_test = x[:num_train_samples], x[num_train_samples:]
    y_train, y_test = y[:num_train_samples], y[num_train_samples:]

    # Build the SimpleRNN model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(40, activation='tanh', return_sequences=True, input_shape=(4, 15)),
        tf.keras.layers.SimpleRNN(40, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

    # Define a learning rate schedule within the optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=num_epochs_to_decay,  # OPP
        decay_rate=1  # OPP
    )

    # Use the optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model with the optimizer
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print('Test Loss:', loss)

    # Make predictions
    y_pred = model.predict(x_test).squeeze()

    # Denormalize the data
    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)
    print("Zero array shape:", np.zeros((y_pred.shape[0], data.shape[1] - 1)).shape)
    y_test_actual = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], data.shape[1] - 1))], axis=1))[:, 0]
    y_pred_actual = scaler.inverse_transform(
        np.concatenate([y_pred.reshape(-1, 1), np.zeros((y_pred.shape[0], data.shape[1] - 1))], axis=1))[:, 0]

    # Plot the de-normalized predicted and actual values
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_actual, label='Actual Prices', color='blue')
    plt.plot(y_pred_actual, label='Predicted Prices', color='red', linestyle='dashed')
    plt.title('Expected vs Actual Closing Prices (CVX RNN)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
