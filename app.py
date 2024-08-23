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

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file):
    try:
        df = pd.read_csv(file, parse_dates=['Date'])
        df = add_technical_indicators(df)
        df = df[df['Volume'] != 0]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Add technical indicators
def add_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    return df.dropna()

# Compute RSI
def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Preprocess data for model
@st.cache_data
def preprocess_for_model(_df):
    features = ['Open', 'Low', 'High', 'Volume']
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(_df[features])
    y = scaler_y.fit_transform(_df[['Close']])
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y, scaler_X, scaler_y, _df['Date']

# Create and train model
@st.cache_resource
def create_and_train_model(X_train, y_train, lstm_units, dropout_rate, batch_size, epochs, optimizer, loss_function):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units // 2, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units // 4, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer=optimizer, loss=loss_function)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Main app
def main():
    st.title('Stock Price Prediction App')
    st.markdown("""
    This app predicts stock prices using a Long Short-Term Memory (LSTM) neural network model.
    Upload a CSV file with columns: Date, Open, High, Low, Close, Adj Close, Volume.
    """)

    # Sidebar for hyperparameters
    st.sidebar.header('Hyperparameters')
    lstm_units = st.sidebar.slider('LSTM Units', 16, 256, 128, 16)
    dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.2, 0.05)
    batch_size = st.sidebar.slider('Batch Size', 16, 128, 32, 16)
    epochs = st.sidebar.slider('Epochs', 50, 300, 150, 50)
    train_percentage = st.sidebar.slider('Training Data Percentage', 50, 90, 80, 5)
    optimizer = st.sidebar.selectbox('Select Optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamax'])
    loss_function = st.sidebar.selectbox('Select Loss Function', ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        if df is not None:
            st.subheader('Descriptive Statistics')
            st.write(df.describe())

            with st.spinner('Processing data and training model...'):
                X, y, scaler_X, scaler_y, dates = preprocess_for_model(df)
                split = int((train_percentage / 100) * len(X))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = create_and_train_model(X_train, y_train, lstm_units, dropout_rate, batch_size, epochs, optimizer, loss_function)
                train_predictions = model.predict(X_train)
                test_predictions = model.predict(X_test)

            # Inverse transform predictions
            train_predictions = scaler_y.inverse_transform(train_predictions)
            test_predictions = scaler_y.inverse_transform(test_predictions)
            y_train_inverse = scaler_y.inverse_transform(y_train)
            y_test_inverse = scaler_y.inverse_transform(y_test)

            # Calculate metrics
            rmse_train = np.sqrt(mean_squared_error(y_train_inverse, train_predictions))
            mae_train = mean_absolute_error(y_train_inverse, train_predictions)
            r2_train = r2_score(y_train_inverse, train_predictions)
            rmse_test = np.sqrt(mean_squared_error(y_test_inverse, test_predictions))
            mae_test = mean_absolute_error(y_test_inverse, test_predictions)
            r2_test = r2_score(y_test_inverse, test_predictions)

            # Display metrics
            st.write(f"Training RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}, R²: {r2_train:.2f}")
            st.write(f"Testing RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}, R²: {r2_test:.2f}")

            # Plot results
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(x=dates[:split], y=y_train_inverse.flatten(), mode='lines', name='Actual Train Prices'))
            fig_train.add_trace(go.Scatter(x=dates[:split], y=train_predictions.flatten(), mode='lines', name='Predicted Train Prices'))
            fig_train.update_layout(title='Training Data: Actual vs Predicted Close Prices', xaxis_title='Date', yaxis_title='Close Price')
            st.plotly_chart(fig_train)

            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=dates[split:], y=y_test_inverse.flatten(), mode='lines', name='Actual Test Prices'))
            fig_test.add_trace(go.Scatter(x=dates[split:], y=test_predictions.flatten(), mode='lines', name='Predicted Test Prices'))
            fig_test.update_layout(title='Testing Data: Actual vs Predicted Close Prices', xaxis_title='Date', yaxis_title='Close Price')
            st.plotly_chart(fig_test)

if __name__ == "__main__":
    main()
