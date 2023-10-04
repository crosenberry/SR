# ANN vs RNN Stock Prediction Comparison
Caden Rosenberry\
Senior Research and Development\
Capstone Project

## Purpose
The purpose of this project is to compare the performance of an Artificial Neural Network (ANN) and a Recurrent Neural Network (RNN) in predicting stock prices. The ANN and RNN will be trained on the same data and then tested on the same data. The performance of the two models will be compared to determine which model is better at predicting stock prices.

## Hypothesis
The RNN-based model will perform better than the ANN-based model and will lead to improve prediction accuracy of Chevron’s/Exxon’s stock percent change 

## Data
This project uses historical stock data for Exxon, Chevron, S&P500 and Crude Oil Prices including:
* Open
* High
* Low
* Close
* Volume

## Requirements
The requirements for this project are as follows:\
* Tensorflow
* Pandas
* Keras
* Numpy
* Scikit-learn
* Yfinance
* Matplotlib 

\
To install these packages, run the following command in the terminal:\
pip install -r requirements.txt

## Usage
To run the project, run the main.py file\
It will prompt with the following options:\
```Chevron or Exxon?:``` Enter Exxon if want to use Exxon data or Chevron for Chevron data\
It will then prompt with the following options:\
```ANN or RNN?:``` Enter ANN if want to use ANN model or RNN for RNN model


