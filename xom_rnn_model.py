import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters


def generate_exxon_rnn(start_dates, end_dates, seed):
    seed_value = seed
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    xom = yf.Ticker('XOM')
    xom_data = xom.history(start=start_dates, end=end_dates)
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(start=start_dates, end=end_dates)
    cl = yf.Ticker('CL=F')
    cl_data = cl.history(start=start_dates, end=end_dates)

    # Combine the data
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

        'XOM_Volume': xom_data['Volume'].values,
        'SP500_Volume': sp500_data['Volume'].values,
        'Oil_Volume': cl_data['Volume'].values
    })

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset
    sequence_length = 4
    x, y = [], []

    for i in range(len(data_scaled) - sequence_length):
        x.append(data_scaled[i:i + sequence_length])
        close_price_today = data_scaled[i + sequence_length][0]
        close_price_prev = data_scaled[i + sequence_length - 1][0]
        percent_change = (close_price_today - close_price_prev) / close_price_prev if close_price_prev != 0 else 0
        y.append(percent_change)

    x, y = np.array(x), np.array(y)
    num_train_samples = int(0.9 * len(x))
    x_train, x_test = x[:num_train_samples], x[num_train_samples:]
    y_train, y_test = y[:num_train_samples], y[num_train_samples:]

    test_dates = xom_data.index.to_series().iloc[num_train_samples + sequence_length:]
    def build_model(hp: HyperParameters):
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(hp.Int('rnn_units_1', min_value=20, max_value=60, step=10),
                                      activation='tanh', return_sequences=True,
                                      input_shape=(sequence_length, 15)),
            tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)),
            tf.keras.layers.SimpleRNN(hp.Int('rnn_units_2', min_value=20, max_value=60, step=10),
                                      activation='tanh'),
            tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)),
            tf.keras.layers.Dense(1)
        ])

        # Define a learning rate within the optimizer
        lr = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='mse')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,  # number of different hyperparameter combinations to test
        executions_per_trial=3,
        directory=f'tuning',
        project_name=f'XOM_RNN - {seed_value}'
    )

    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=70, batch_size=32, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the model
    history = best_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)  # OPP

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
    bin_edges = [-np.inf, -0.01, -0.001, 0.001, 0.01, np.inf]
    bin_labels = ["Strong Decrease", "Decrease", "Stable", "Increase", "Strong Increase"]
    results['Actual Bin'] = pd.cut(results['Actual Percent Change'], bin_edges, labels=bin_labels)
    results['Predicted Bin'] = pd.cut(results['Predicted Percent Change'], bin_edges, labels=bin_labels)

    # Plot the distribution of actual vs predicted bins
    bin_counts_actual = results['Actual Bin'].value_counts().sort_index()
    bin_counts_predicted = results['Predicted Bin'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    bin_counts_actual.plot(kind='bar', color='blue', alpha=0.6, label='Actual')
    bin_counts_predicted.plot(kind='bar', color='red', alpha=0.6, label='Predicted')
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.title('Distribution of Actual vs Predicted Percent Changes')
    plt.legend()
    plt.show()

    # You might also want to plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    generate_exxon_rnn('2018-04-01', '2019-05-05', 44)
