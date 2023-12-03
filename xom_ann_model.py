import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters
from sklearn.metrics import classification_report, accuracy_score


# This file is for the ANN model for Exxon
def generate_exxon_ann(start_dates, end_dates, seed):
    seed_value = seed
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    xom = yf.Ticker('XOM')
    xom_data = xom.history(start=start_dates, end=end_dates)

    # Combine the data of the three stocks
    data = pd.DataFrame({
        'XOM_Open': xom_data['Open'].values,
        'XOM_Close': xom_data['Close'].values,
        'XOM_High': xom_data['High'].values,
        'XOM_Low': xom_data['Low'].values,
        'XOM_Volume': xom_data['Volume'].values,
    })
    print(data.head())

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset
    sequence_length = 4  # Length of input sequence (lookback)
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

    # Convert percent changes to categorical bins
    bin_edges = [-np.inf, -0.05, -0.015, 0.015, 0.05, np.inf]
    bin_labels = ["Strong Decrease (< -5%)",
                  "Decrease (-5% to -1.5%)",
                  "Stable (-1.5% to 1.5%)",
                  "Increase (1.5% to 5%)",
                  "Strong Increase (> 5%)"]
    y_binned = np.digitize(y, bins=bin_edges) - 1
    y_categorical = tf.keras.utils.to_categorical(y_binned, num_classes=len(bin_labels))

    # Define our training and testing data
    num_train_samples = int(0.9 * len(x))
    x_train, x_test = x[:num_train_samples], x[num_train_samples:]
    y_train, y_test = y_categorical[:num_train_samples], y_categorical[num_train_samples:]

    test_dates = xom_data.index.to_series().iloc[num_train_samples + sequence_length:]

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Wrap ANN model in a function for KerasTuner
    def build_model(hp: HyperParameters):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32),
                                  activation='tanh',
                                  input_shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(5, activation='softmax')  # Output layer for 5 bins
        ])

        lr = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Initialize the tuner and pass the `build_model` function
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,  # number of different hyperparameter combinations to test
        executions_per_trial=5,
        directory=f'exxon_ann_tuning',
        project_name=f'XOM_ANN - {seed_value}'
    )

    # Find the optimal hyperparameters
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model
    history = best_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    train_loss, train_accuracy = best_model.evaluate(x_train, y_train)
    print('Train Loss:', train_loss)
    print('Train Accuracy:', train_accuracy)
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

    # Make predictions
    y_pred_prob = best_model.predict(x_test)
    y_pred_binned = np.argmax(y_pred_prob, axis=1)
    y_test_binned = np.argmax(y_test, axis=1)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test_binned, y_pred_binned)
    print(f'Accuracy: {accuracy:.2f}')

    class_report = classification_report(y_test_binned, y_pred_binned, target_names=bin_labels, zero_division=0)
    print("Classification Report:")
    print(class_report)

    # Calculate bin counts for actual and predicted
    bin_counts_actual = np.bincount(y_test_binned, minlength=len(bin_labels))
    bin_counts_predicted = np.bincount(y_pred_binned, minlength=len(bin_labels))

    # Plotting the distribution of actual vs predicted bins
    x = np.arange(len(bin_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, bin_counts_actual, width, label='Actual', color='blue')
    rects2 = ax.bar(x + width / 2, bin_counts_predicted, width, label='Predicted', color='red')

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('Bins')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Actual vs Predicted Bins (XOM ANN- Seed {seed_value})')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)  # Rotate labels for better readability
    ax.legend()

    # Function to add count above each bar
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

    # Prepare data for statistical analysis
    epochs = range(1, 51)  # Assuming 50 epochs
    history_data = {
        'Epoch': epochs,
        'Train Accuracy': history.history['accuracy'],
        'Train Loss': history.history['loss'],
        'Validation Accuracy': history.history['val_accuracy'],
        'Validation Loss': history.history['val_loss']
    }

    # Convert the dictionary to a DataFrame
    history_df = pd.DataFrame(history_data)

    # Save the history DataFrame to a CSV file
    history_file_name = f'exxon_ann_history_seed_{seed_value}.csv'
    history_df.to_csv(history_file_name, index=False)
    print(f'History data saved to {history_file_name}')

# learning rate
if __name__ == '__main__':
    generate_exxon_ann('2018-04-01', '2019-05-05', 194)
