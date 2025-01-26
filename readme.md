# Bitcoin Price Prediction
(Please don't use this for trading)
This project predicts **_today's_** price of Bitcoin using an LSTM (Long Short-Term Memory) neural network model. The model is trained using historical Bitcoin price data, and predictions are visualized with an interactive candlestick chart.

## Overview

- **Data Source:** The project uses historical Bitcoin price data from a CSV file named `bitcoin.csv`.
- **Model:** An LSTM neural network is used to predict the future price based on past price movements.
- **Visualization:** Interactive candlestick charts are generated using Plotly to visualize the actual vs. predicted prices.

# Bitcoin Price Prediction

## Files

- **bitcoin.csv**: The CSV file containing the historical Bitcoin data. It must include the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- **bitcoin_price_model.h5**: The file where the trained LSTM model is saved. If it doesn't exist, a new model will be trained and saved to this file.

## How It Works

1. **Data Loading and Preprocessing**:  
   The script reads the `bitcoin.csv` file, converts the `Date` column to a datetime format, and sets it as the index. It then separates the last data point to compare with the predicted value.

2. **Normalization**:  
   The `Adj Close` prices are normalized using `MinMaxScaler` to scale the data between 0 and 1.

3. **Model Training**:  
   - The model uses a sequence of 60 previous days to predict the next day’s price.
   - If a pre-trained model (`bitcoin_price_model.h5`) is found, it is loaded. Otherwise, a new LSTM model is trained with the existing data.

4. **Prediction**:  
   The model predicts the price for the next day, which is compared with the actual last data point.

5. **Visualization**:  
   A candlestick chart is generated to visualize the historical price data. The actual and predicted prices are added as points on the chart.

   ![image](https://github.com/user-attachments/assets/01342b7a-73c4-4c07-a9ee-0070577a2d3f)
   ![image](https://github.com/user-attachments/assets/65dec69a-1a7b-4dd6-921d-693b917b163a)

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
run working.py






