# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:34:25 2024

@author: Tayyab
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go

# Load sample data from a CSV file
@st.cache_data
def load_sample_data():
    return pd.read_csv('AAPL.csv', parse_dates=['Date'])

# Title and description
st.title('Stock Price Prediction App')
st.markdown("""
This app predicts stock prices using a Long Short-Term Memory (LSTM) neural network model. 
LSTM is a type of recurrent neural network (RNN) that is well-suited for time series data, 
such as stock prices, because it can learn patterns over time.

### How to Use This App
1. **Upload a CSV File**: The file should contain the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
2. **Adjust Hyperparameters**: Use the sidebar to modify model settings and observe how they affect predictions.
3. **View Predictions**: The app will display actual vs. predicted prices for both training and testing datasets.
""")

# Display example data rows
sample_df = load_sample_data()
st.markdown('Example data rows:')
st.dataframe(sample_df.head())

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Sidebar for hyperparameters
st.sidebar.header('Hyperparameters')
st.sidebar.markdown("""
- **LSTM Units**: Number of units in each LSTM layer. More units can capture more complex patterns. 
  [Learn more](https://en.wikipedia.org/wiki/Long_short-term_memory)
- **Dropout Rate**: Fraction of units to drop for regularization. Helps prevent overfitting. 
  [Learn more](https://en.wikipedia.org/wiki/Dropout_(neural_networks))
- **Batch Size**: Number of samples per gradient update. Affects training speed and stability.
- **Epochs**: Number of times the model will iterate over the entire dataset during training.
- **Training Data Percentage**: Proportion of data used for training vs. testing.
- **Optimizer**: Algorithm used to update model weights. Different optimizers can affect convergence speed.
- **Loss Function**: Metric used to evaluate model performance during training.
""")

lstm_units = st.sidebar.slider('LSTM Units', min_value=16, max_value=256, value=128, step=16)
dropout_rate = st.sidebar.slider('Dropout Rate', min_value=0.0, max_value=0.5, value=0.2, step=0.05)
batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)
epochs = st.sidebar.slider('Epochs', min_value=50, max_value=300, value=150, step=50)

# Percentage split for training and testing data
train_percentage = st.sidebar.slider('Training Data Percentage', min_value=50, max_value=90, value=80, step=5)

# Dropdown for optimizer selection
optimizer_options = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamax']
selected_optimizer = st.sidebar.selectbox('Select Optimizer', optimizer_options)

# Dropdown for loss function selection
loss_function_options = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
selected_loss_function = st.sidebar.selectbox('Select Loss Function', loss_function_options)

# Function to add technical indicators
def add_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df = df.dropna()  # Drop rows with NaN values after adding indicators
    return df

# Function to compute RSI
def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to preprocess the data
def preprocess_data(df):
    df = add_technical_indicators(df)
    df = df[df['Volume'] != 0]
    features = ['Open', 'Low', 'High', 'Volume']
    
    # Scale features
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(df[features])
    
    # Scale target variable separately
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(df[['Close']])  # Scale the target as well
    
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM
    return X, y, scaler_X, scaler_y, df['Date']

# Function to create and train the LSTM model
def create_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units // 2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units // 4, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))

    model.compile(optimizer=selected_optimizer, loss=selected_loss_function)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)  # Increased epochs
    return model

# Predict and display results
if uploaded_file is not None or sample_df is not None:
    # Use uploaded file if available, otherwise use sample data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    else:
        df = sample_df

    # Display descriptive statistics
    st.subheader('Descriptive Statistics')
    st.write(df.describe())

    # Inform user that the model is being trained
    with st.spinner('Training the model, please wait...'):
        # Preprocess the data
        X, y, scaler_X, scaler_y, dates = preprocess_data(df)

        # Split the data into training and test sets
        split = int((train_percentage / 100) * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Display the amount of data used for training
        st.write(f"Training Data Size: {len(X_train)} samples")

        # Create and train the model
        model = create_and_train_model(X_train, y_train)

        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

    # Inverse transform the predictions
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_inverse = scaler_y.inverse_transform(y_train)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    # Calculate RMSE, MAE, and R-squared
    rmse_train = np.sqrt(mean_squared_error(y_train_inverse, train_predictions))
    mae_train = mean_absolute_error(y_train_inverse, train_predictions)
    r2_train = r2_score(y_train_inverse, train_predictions)

    rmse_test = np.sqrt(mean_squared_error(y_test_inverse, test_predictions))
    mae_test = mean_absolute_error(y_test_inverse, test_predictions)
    r2_test = r2_score(y_test_inverse, test_predictions)

    # Display metrics
    st.write(f"Training RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}, R²: {r2_train:.2f}")
    st.write(f"Testing RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}, R²: {r2_test:.2f}")

    # Plotting training predictions
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=dates[:split], y=y_train_inverse.flatten(), mode='lines', name='Actual Train Prices'))
    fig_train.add_trace(go.Scatter(x=dates[:split], y=train_predictions.flatten(), mode='lines', name='Predicted Train Prices'))

    fig_train.update_layout(
        title='Training Data: Actual vs Predicted Close Prices',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x'
    )

    st.plotly_chart(fig_train)

    # Plotting test predictions
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=dates[split:], y=y_test_inverse.flatten(), mode='lines', name='Actual Test Prices'))
    fig_test.add_trace(go.Scatter(x=dates[split:], y=test_predictions.flatten(), mode='lines', name='Predicted Test Prices'))

    fig_test.update_layout(
        title='Testing Data: Actual vs Predicted Close Prices',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x'
    )

    st.plotly_chart(fig_test)