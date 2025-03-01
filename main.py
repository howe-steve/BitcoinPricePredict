import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load the CSV data
df = pd.read_csv('bitcoin.csv')

# Convert 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Prepare the data for candlestick chart and model
candlestick_data = df[['Open', 'High', 'Low', 'Close']]
prices = df['Adj Close'].values
prices = prices.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Prepare the training data
look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_prices)):
    X.append(scaled_prices[i - look_back:i, 0])
    y.append(scaled_prices[i, 0])
X, y = np.array(X), np.array(y)

# Reshape X to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Check if model file exists
model_file = 'bitcoin_price_model.h5'
try:
    model = tf.keras.models.load_model(model_file)
    print("Model loaded successfully.")
except IOError:
    print("No saved model found. Training a new model.")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=50)
    model.save(model_file)
    print("Model trained and saved.")

# Function for incremental training
def incremental_training(new_data, model, scaler, look_back=60, epochs=10, batch_size=1):
    # Prepare the new data
    prices = new_data['Adj Close'].values
    prices = prices.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)

    # Prepare the new training data
    X_new, y_new = [], []
    for i in range(look_back, len(scaled_prices)):
        X_new.append(scaled_prices[i - look_back:i, 0])
        y_new.append(scaled_prices[i, 0])
    X_new, y_new = np.array(X_new), np.array(y_new)
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

    # Recreate the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Recompile the model with the new optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model incrementally
    model.fit(X_new, y_new, batch_size=batch_size, epochs=epochs)

    # Save the updated model
    model.save(model_file)
    print("Model updated and saved.")

# Example usage of incremental training
new_data = pd.read_csv('new_bitcoin_data.csv')
new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data.set_index('Date', inplace=True)
incremental_training(new_data, model, scaler)

# Prepare the data for prediction
last_scaled_data = scaled_prices[-look_back:]
last_scaled_data = np.reshape(last_scaled_data, (1, look_back, 1))

# Predict the last data point
predicted_price_scaled = model.predict(last_scaled_data)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# Compare with actual last data point
actual_last_price = df['Adj Close'].values[-1]
print(f"Predicted Price: {predicted_price[0][0]}")
print(f"Actual Last Price: {actual_last_price}")

# Create an interactive candlestick chart using Plotly
candlestick = go.Figure(data=[go.Candlestick(x=candlestick_data.index,
                                             open=candlestick_data['Open'],
                                             high=candlestick_data['High'],
                                             low=candlestick_data['Low'],
                                             close=candlestick_data['Close'])])

# Add the predicted and actual last price points to the chart
candlestick.add_trace(go.Scatter(x=[candlestick_data.index[-1]],
                                 y=[predicted_price[0][0]],
                                 mode='markers',
                                 marker=dict(color='blue', size=10),
                                 name='Predicted Price'))

candlestick.add_trace(go.Scatter(x=[candlestick_data.index[-1]],
                                 y=[actual_last_price],
                                 mode='markers',
                                 marker=dict(color='red', size=10),
                                 name='Actual Last Price'))

# Update layout for better visualization
candlestick.update_layout(title='Bitcoin Price Prediction with Candlestick Chart',
                          xaxis_title='Date',
                          yaxis_title='Price (USD)',
                          xaxis_rangeslider_visible=False)

# Show the interactive chart
candlestick.show()
