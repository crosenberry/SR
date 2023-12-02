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

    # Combine the data
    data = pd.DataFrame({
        'XOM_Close': xom_data['Close'].values,
        'XOM_Open': xom_data['Open'].values,
        'XOM_High': xom_data['High'].values,
        'XOM_Low': xom_data['Low'].values,
        'XOM_Volume': xom_data['Volume'].values
    })

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset
    sequence_length = 4
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

    # Convert percent changes to categorical bins
    bin_edges = [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
    bin_labels = ["Strong Decrease (< -5%)",
                  "Decrease (-5% to -1%)",
                  "Stable (-1% to 1%)",
                  "Increase (1% to 5%)",
                  "Strong Increase (> 5%)"]
    y_binned = np.digitize(y, bins=bin_edges) - 1
    y_categorical = tf.keras.utils.to_categorical(y_binned, num_classes=len(bin_labels))

    num_train_samples = int(0.9 * len(x))
    x_train, x_test = x[:num_train_samples], x[num_train_samples:]
    y_train, y_test = y_categorical[:num_train_samples], y_categorical[num_train_samples:]
    test_dates = xom_data.index.to_series().iloc[num_train_samples + sequence_length:]

  # RNN model architecture
    def build_model(hp: HyperParameters):
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(hp.Int('rnn_units_1', min_value=20, max_value=60, step=10),
                                      activation='tanh', return_sequences=True,
                                      input_shape=(sequence_length, 5)),
            tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)),
            tf.keras.layers.SimpleRNN(hp.Int('rnn_units_2', min_value=20, max_value=60, step=10),
                                      activation='tanh'),
            tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)),
            tf.keras.layers.Dense(len(bin_labels), activation='softmax')  # Output layer for bins
        ])

        # Define a learning rate within the optimizer
        lr = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.001)
        print("Learning rate:", lr)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=15,  # number of different hyperparameter combinations to test
        executions_per_trial=3,
        directory=f'exxon_rnn_tuning',
        project_name=f'XOM_RNN - {seed_value}'
    )

    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=70, batch_size=32, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the model
    trained_model = best_model.fit(x_train, y_train, epochs=60, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

    # Make predictions and categorize into bins
    y_pred_prob = best_model.predict(x_test)
    y_pred_binned = np.argmax(y_pred_prob, axis=1)
    y_test_binned = np.argmax(y_test, axis=1)

    # Plotting the distribution of actual vs predicted bins
    results = pd.DataFrame({
        'Actual Bin': y_test_binned,
        'Predicted Bin': y_pred_binned
    })

    bin_counts_actual = np.bincount(results['Actual Bin'], minlength=len(bin_labels))
    bin_counts_predicted = np.bincount(results['Predicted Bin'], minlength=len(bin_labels))

    # Create the plot
    x = np.arange(len(bin_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, bin_counts_actual, width, label='Actual', color='blue')
    rects2 = ax.bar(x + width / 2, bin_counts_predicted, width, label='Predicted', color='red')

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('Bins')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Actual vs Predicted Bins (XOM RNN- Seed {seed_value})')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)  # Rotate labels for better readability
    ax.legend()

    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == '__main__':
    generate_exxon_rnn('2018-04-01', '2019-05-05', 87)