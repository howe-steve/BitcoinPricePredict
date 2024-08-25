# Bitcoin Price Prediction

This project predicts the future price of Bitcoin using an LSTM (Long Short-Term Memory) neural network model. The model is trained using historical Bitcoin price data, and predictions are visualized with an interactive candlestick chart.

## Overview

- **Data Source:** The project uses historical Bitcoin price data from a CSV file named `bitcoin.csv`.
- **Model:** An LSTM neural network is used to predict the future price based on past price movements.
- **Visualization:** Interactive candlestick charts are generated using Plotly to visualize the actual vs. predicted prices.

## Dependencies

The following Python packages are required to run this project:

- pandas
- numpy
- tensorflow
- scikit-learn
- plotly

You can install these dependencies using the following command:

```bash
pip install pandas numpy tensorflow scikit-learn plotly
