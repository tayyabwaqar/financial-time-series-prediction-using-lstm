# Financial Time Series Prediction App

This app predicts stock prices using a Long Short-Term Memory (LSTM) neural network model. LSTM is a type of recurrent neural network (RNN) that is well-suited for time series data, such as stock prices, because it can learn patterns over time.

## Live Demo

You can try the live demo of the application at the following link: 
[Live Demo - LSTM Time Series Prediction App](https://interactive-stock-price-predictor.streamlit.app/)

## How to Use

- Upload a CSV File: The file should contain the required columns mentioned above.
- Adjust Hyperparameters: Use the sidebar to modify model settings and observe how they affect predictions.

## Features

- Upload CSV files containing stock market data.
- Adjust hyperparameters for the LSTM model.
- View actual vs. predicted stock prices with interactive visualizations.
- Display descriptive statistics of the dataset.

## Sample Data
The app comes with a sample CSV file (sample_stock_data.csv) that contains the following columns:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

You can view example data rows in the app to understand the expected format.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-app.git
   cd stock-price-prediction-app

2. **Install the required packages**:

   To run this app, you need to have the following Python packages installed:

  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `plotly`

   You can install the required packages using the following command
    
    pip install -r requirements.txt

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py

4. **Open your web browser and navigate to http://localhost:8501 to view the app.**
   
## License
This project is licensed under the MIT License - see the LICENSE file for details.
