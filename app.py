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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, SimpleRNN
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Function to guess date format
def guess_date_format(date_string):
    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%m.%d.%Y",
    ]
    for date_format in date_formats:
        try:
            datetime.strptime(date_string, date_format)
            return date_format
        except ValueError:
            continue
    return None

# Function to fetch real-time stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to add technical indicators
def add_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df = df.dropna()
    return df

# Function to compute RSI
def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to preprocess the data
def preprocess_data(df, features):
    df = add_technical_indicators(df)
    df = df[df['Volume'] != 0]
    
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(df[features])
    
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(df[['Close']])
    
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y, scaler_X, scaler_y, df['Date']

# Function to create and train the model
def create_and_train_model(X_train, y_train, model_type, lstm_units, dropout_rate, optimizer, loss_function, epochs, batch_size):
    if model_type == 'LSTM':
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=lstm_units // 2, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=lstm_units // 4, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=1)
        ])
    elif model_type == 'GRU':
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            GRU(units=lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            GRU(units=lstm_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=1)
        ])
    elif model_type == 'SimpleRNN':
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            SimpleRNN(units=lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            SimpleRNN(units=lstm_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=1)
        ])

    model.compile(optimizer=optimizer, loss=loss_function)
    
    with st.spinner('Training the model, please wait...'):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    
    return model, history

def create_candlestick_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        subplot_titles=('Candlestick', 'Volume'),
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'), row=1, col=1)

    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=2, col=1)
    
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'), row=1, col=1, yaxis="y2")
        fig.update_layout(yaxis2=dict(title="RSI", overlaying="y", side="right"))

    fig.update_layout(title='Stock Price and Volume Over Time',
                      xaxis_rangeslider_visible=False)
    return fig

# Function to get table download link
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
    return href

# Streamlit app
st.title('Advanced Stock Price Prediction App')

# Sidebar for input parameters
st.sidebar.header('Input Parameters')

# Data source selection
data_source = st.sidebar.radio("Select Data Source", ('Upload CSV', 'Fetch Real-time Data'))

if data_source == 'Upload CSV':
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
else:
    ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")
    start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End date", datetime.now())

# Model parameters
model_type = st.sidebar.selectbox('Select Model Type', ['LSTM', 'GRU', 'SimpleRNN'])
lstm_units = st.sidebar.slider('LSTM/GRU/RNN Units', min_value=16, max_value=256, value=128, step=16)
dropout_rate = st.sidebar.slider('Dropout Rate', min_value=0.0, max_value=0.5, value=0.2, step=0.05)
batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)
epochs = st.sidebar.slider('Epochs', min_value=50, max_value=300, value=150, step=50)
train_percentage = st.sidebar.slider('Training Data Percentage', min_value=50, max_value=90, value=80, step=5)

optimizer_options = {'adam': Adam(), 'rmsprop': RMSprop(), 'sgd': SGD()}
selected_optimizer = st.sidebar.selectbox('Select Optimizer', list(optimizer_options.keys()))
optimizer = optimizer_options[selected_optimizer]

loss_function_options = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
selected_loss_function = st.sidebar.selectbox('Select Loss Function', loss_function_options)

features = st.sidebar.multiselect('Select Features', 
                                  ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'RSI', 'MACD', 'Signal'], 
                                  default=['Open', 'High', 'Low', 'Close', 'Volume'])

# Load data
if data_source == 'Upload CSV' and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    date_format = guess_date_format(df['Date'].iloc[0])
    if date_format:
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    else:
        st.error("Unable to determine the date format. Please ensure your 'Date' column is in a recognizable format.")
        st.stop()
elif data_source == 'Fetch Real-time Data':
    df = get_stock_data(ticker, start_date, end_date)
    if df is None:
        st.stop()
else:
    st.warning("Please upload a CSV file or fetch real-time data.")
    st.stop()

# Add technical indicators
df = add_technical_indicators(df)

# Display raw data and candlestick chart
st.subheader('Raw Data')
st.write(df.head())

st.subheader('Candlestick Chart')
st.plotly_chart(create_candlestick_chart(df))

# Preprocess data
X, y, scaler_X, scaler_y, dates = preprocess_data(df, features)


# Split data
split = int((train_percentage / 100) * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create and train model
model, history = create_and_train_model(X_train, y_train, model_type, lstm_units, dropout_rate, optimizer, selected_loss_function, epochs, batch_size)

# Make predictions
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
st.subheader('Model Performance Metrics')
col1, col2 = st.columns(2)
with col1:
    st.write("Training Set:")
    st.write(f"RMSE: {rmse_train:.2f}")
    st.write(f"MAE: {mae_train:.2f}")
    st.write(f"R²: {r2_train:.2f}")
with col2:
    st.write("Testing Set:")
    st.write(f"RMSE: {rmse_test:.2f}")
    st.write(f"MAE: {mae_test:.2f}")
    st.write(f"R²: {r2_test:.2f}")

# Plot training history
st.subheader('Training History')
fig_history = go.Figure()
fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
fig_history.update_layout(title='Model Training History', xaxis_title='Epoch', yaxis_title='Loss')
st.plotly_chart(fig_history)

# Plot predictions
st.subheader('Price Predictions')
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=dates[:split], y=y_train_inverse.flatten(), mode='lines', name='Actual Train'))
fig_pred.add_trace(go.Scatter(x=dates[:split], y=train_predictions.flatten(), mode='lines', name='Predicted Train'))
fig_pred.add_trace(go.Scatter(x=dates[split:], y=y_test_inverse.flatten(), mode='lines', name='Actual Test'))
fig_pred.add_trace(go.Scatter(x=dates[split:], y=test_predictions.flatten(), mode='lines', name='Predicted Test'))
fig_pred.update_layout(title='Actual vs Predicted Stock Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_pred)

# Export predictions
predictions_df = pd.DataFrame({
    'Date': dates[split:],
    'Actual': y_test_inverse.flatten(),
    'Predicted': test_predictions.flatten()
})
st.subheader('Export Predictions')
st.markdown(get_table_download_link(predictions_df), unsafe_allow_html=True)

# Future predictions
st.subheader('Future Price Predictions')
future_days = st.number_input('Number of days to predict', min_value=1, max_value=30, value=7)

last_sequence = X_test[-1]
future_predictions = []

for _ in range(future_days):
    future_pred = model.predict(last_sequence.reshape(1, 1, -1))
    future_predictions.append(future_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[0, -1, -1] = future_pred

future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=future_days)

fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=dates, y=scaler_y.inverse_transform(y).flatten(), mode='lines', name='Historical'))
fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Prediction'))
fig_future.update_layout(title='Future Stock Price Predictions', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_future)

st.markdown("""
## How to Use This App
1. **Select Data Source**: Choose between uploading a CSV file or fetching real-time data from Yahoo Finance.
2. **Upload CSV**: If you choose to upload a CSV file, make sure it contains the columns: `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.
3. **Fetch Real-time Data**: If you select this option, enter the stock ticker (e.g., AAPL) and specify the date range for fetching historical data.
4. **Adjust Model Parameters**: Use the sidebar to set hyperparameters for the model, including the type of model (LSTM, GRU, or Simple RNN), number of units, dropout rate, batch size, and epochs.
5. **Select Features**: Choose which features to include in the model. You can select from the available technical indicators.
6. **View Results**: After training, the app will display the model's performance metrics, training history, actual vs. predicted prices, and allow you to download the predictions.
7. **Future Predictions**: Specify the number of future days to predict and view the forecasted stock prices.

### Additional Features
- The app includes visualizations for candlestick charts, volume, and technical indicators like RSI and MACD.
- You can compare actual and predicted prices for both training and testing datasets.
- The app allows you to download the predictions as a CSV file for further analysis.

### Future Enhancements
- Consider adding more advanced features like automated hyperparameter tuning, ensemble methods, and improved error handling.
- You could also implement user authentication to save user preferences and analyses.""")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("### Welcome to the Advanced Stock Price Prediction App!")
