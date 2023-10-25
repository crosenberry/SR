import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# This file is for the ANN model for Exxon
def generate_exxon_ann(start_dates, end_dates):
    seed_value = 44
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    num_epochs_to_decay = 10
    xom = yf.Ticker('XOM')
    xom_data = xom.history(start=start_dates, end=end_dates)
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(start=start_dates, end=end_dates)
    print(sp500_data.head())
    cl = yf.Ticker('CL=F')
    cl_data = cl.history(start=start_dates, end=end_dates)
    print(cl_data.head())

    # Combine the data of the three stocks
    data = pd.DataFrame({

        'XOM_Open': xom_data['Open'].values,
        'SP500_Open': sp500_data['Open'].values,
        'Oil_Open': cl_data['Open'].values,

        'XOM_Close': xom_data['Close'].values,
        'SP500_Close': sp500_data['Close'].values,
        'Oil_Close': cl_data['Close'].values,

        'XOM_High': xom_data['High'].values,
        'SP500_High': sp500_data['High'].values,
        'Oil_High': cl_data['High'].values,

        'XOM_Low': xom_data['Low'].values,
        'SP500_Low': sp500_data['Low'].values,
        'Oil_Low': cl_data['Low'].values,

        'XOM_Volume': xom_data['Volume'].values,
        'SP500_Volume': sp500_data['Volume'].values,
        'Oil_Volume': cl_data['Volume'].values
    })
    print(data.head())

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset
    sequence_length = 4  # Length of input sequence
    x, y = [], []

    for i in range(len(data_scaled) - sequence_length):
        x.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length][0])  # Use XOM_Close as the output

    x, y = np.array(x), np.array(y)
    x = x.reshape(x.shape[0], -1)

    # Define our training and testing data. 360 days for training, 40 days for testing.
    num_train_samples = int(0.9 * len(x))
    x_train, x_test = x[:num_train_samples], x[num_train_samples:]
    y_train, y_test = y[:num_train_samples], y[num_train_samples:]

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Build the ANN model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation='tanh', input_shape=(x.shape[1],)),  # Adjusted input shape
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
    model.fit(x_train, y_train, epochs=70, batch_size=40, validation_split=0.2)

    # Evaluate the model
    train_loss = (model.evaluate(x_train, y_train))
    print('Train Loss:', train_loss)
    test_loss = model.evaluate(x_test, y_test)
    print('Test Loss:', test_loss)

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
    plt.title(f'Expected vs Actual Closing Prices (XOM ANN - Seed {seed_value})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    generate_exxon_ann('2018-04-01', '2019-05-05')
