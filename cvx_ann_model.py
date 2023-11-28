import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters


# This file is for the ANN model for Chevron
def generate_chevron_ann(start_dates, end_dates, seed):
    seed_value = seed
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    cvx = yf.Ticker('CVX')
    cvx_data = cvx.history(start=start_dates, end=end_dates)
    print(cvx_data.head())
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
        # Calculate percent change for the closing price
        close_price_today = data_scaled[i + sequence_length][3]  # XOM_Close index after scaling
        close_price_prev = data_scaled[i + sequence_length - 1][3]  # Previous day's XOM_Close
        if close_price_prev != 0:
            percent_change = (close_price_today - close_price_prev) / close_price_prev
            y.append(percent_change)
        else:
            # Handle the case where previous close is zero
            y.append(0)

    x, y = np.array(x), np.array(y)
    x = x.reshape(x.shape[0], -1)

    # Define our training and testing data. 360 days for training, 40 days for testing.
    num_train_samples = int(0.9 * len(x))
    x_train, x_test = x[:num_train_samples], x[num_train_samples:]
    y_train, y_test = y[:num_train_samples], y[num_train_samples:]
    test_dates = cvx_data.index.to_series().iloc[num_train_samples + sequence_length:]

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    def build_model(hp: HyperParameters):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32),
                                  activation='tanh',
                                  input_shape=(x.shape[1],)),
            tf.keras.layers.Dense(1)
        ])

        # Define a learning rate within the optimizer
        lr = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.001),
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        print("Learning rate:", lr)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    # Initialize the tuner and pass the `build_model` function
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,  # number of different hyperparameter combinations to test
        executions_per_trial=3,
        directory=f'chevron_ann_tuning',
        project_name=f'CVX_ANN - {seed_value}'
    )

    # Find the optimal hyperparameters
    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model
    trained_model = best_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = best_model.evaluate(x_test, y_test)
    print('Test Loss:', loss)

    # Make predictions
    y_pred = best_model.predict(x_test).squeeze()

    # Bin the percent changes
    results = pd.DataFrame({
        'Actual Percent Change': y_test,
        'Predicted Percent Change': y_pred
    })
    bin_edges = [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
    bin_labels = ["Strong Decrease", "Decrease", "Stable", "Increase", "Strong Increase"]
    results['Actual Bin'] = pd.cut(results['Actual Percent Change'], bin_edges, labels=bin_labels)
    results['Predicted Bin'] = pd.cut(results['Predicted Percent Change'], bin_edges, labels=bin_labels)

    # Denormalize the data
    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)
    print("Zero array shape:", np.zeros((y_pred.shape[0], data.shape[1] - 1)).shape)

    # Plot the distribution of actual vs predicted bins
    bin_counts_actual = results['Actual Bin'].value_counts().sort_index()
    bin_counts_predicted = results['Predicted Bin'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    bin_counts_actual.plot(kind='bar', color='blue', alpha=0.6, label='Actual')
    bin_counts_predicted.plot(kind='bar', color='red', alpha=0.6, label='Predicted')
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.title(f'Distribution of Actual vs Predicted Percent Changes (CVX ANN - Seed {seed_value}')
    plt.legend()
    plt.show()

    # plot the training and validation loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(trained_model.history['loss'], label='Training Loss')
    # plt.plot(trained_model.history['val_loss'], label='Validation Loss')
    # plt.title(f'Training and Validation Loss Over Epochs (CVX ANN - Seed {seed_value}')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # Plot the actual percent change and the predicted percent change
    if y_pred.size == 0:
        print("y_pred is empty!")
    else:
        # Plot the actual percent change and the predicted percent change
        plt.figure(figsize=(14, 7))
        plt.plot(test_dates, y_test * 100, label='Actual Percent Change', color='blue')  # Scale to percentage
        plt.plot(test_dates, y_pred * 100, label='Predicted Percent Change', color='red',
                 linestyle='dashed')  # Scale to percentage
        plt.title(f'Expected vs Actual Percent Change in Closing Prices (CVX ANN - Seed {seed_value})')
        plt.xlabel('Date')
        plt.ylabel('Percent Change')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    generate_chevron_ann('2018-04-01', '2019-05-05', 405)
