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

# Separate the last data point
# THIS REMOVES THE POINT WE'RE TRYING TO PREDICT FROM THE MODEL
# Toggle to include or exclude the last data point
include_last_data = True

if include_last_data:
    # Extract the last row (last data point)
    last_data = df.iloc[-1:]
else:
    # Extract the last row (last data point) and remove it from the dataset
    last_data = df.iloc[-1:]  # Extract the last row, which will be used for prediction
    df = df.iloc[:-1]  # Remove the last row from the dataset so it is not included in training

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

    # Define the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, batch_size=1, epochs=50)

    # Save the trained model
    model.save(model_file)
    print("Model trained and saved.")

# Prepare the data for prediction
last_scaled_data = scaled_prices[-look_back:]
last_scaled_data = np.reshape(last_scaled_data, (1, look_back, 1))

# Predict the last data point
predicted_price_scaled = model.predict(last_scaled_data)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# Compare with actual last data point
actual_last_price = last_data['Adj Close'].values[0]
print(f"Tomorrow's Predicted Price: {predicted_price[0][0]}")
print(f"Today's Last Price: {actual_last_price}")

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