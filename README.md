# NVIDIA-Stocks-Time-Series-Analysis
Deep Learning project forecasting NVIDIA (NVDA) stock prices using time series analysis


## Project Overview
This project applies deep learning techniques to forecast NVIDIA Corporation's (NVDA) stock price over the next 30 trading days. The goal is to explore how a hybrid CNN-BiLSTM model can capture complex temporal patterns in financial data and produce meaningful short-term predictions.
The dataset contains over 6,500 daily records of NVIDIA stock from 1999 to 2024, sourced from Yahoo Finance. The analysis is focused on the post-2020 period for improved model relevance. An interactive Streamlit dashboard was built to visualize results, explore data, and run forecasts directly from the browser.

## About the Dataset
The dataset was obtained from Yahoo Finance and contains 6,558 daily entries of NVIDIA stock market data spanning from January 1999 to 2024. It includes the following features: Date, Open, High, Low, Close, Adjusted Close, and Volume. For modeling purposes, the dataset was filtered to the post-2020 period (~1,200 trading days), as recent market behavior is more representative of current dynamics. No missing values or duplicates were found in the data.

## Methodology / Project Steps
We loaded the CSV dataset in Python and explored its structure using pandas. The data was cleaned by checking for nulls, duplicates, and date continuity gaps. Outlier analysis was performed using the IQR method across all numeric columns. We conducted exploratory data analysis using histograms, volume charts, and correlation heatmaps to understand price distributions and feature relationships. From the raw OHLCV data, we engineered 19 technical indicators including multi-period returns (1, 3, 5, 10, 20 days), moving average ratios (MA5, MA10, MA20, MA50), rolling volatility measures, RSI, normalized MACD with signal line, Bollinger Band positioning, high-low percentage, open-close percentage, and volume ratio. All features were normalized using MinMaxScaler to the range [-1, 1]. The dataset was split into 72% training, 14% validation, and 14% test sets. We built a hybrid architecture combining a Conv1D layer for short-term pattern extraction, a Bidirectional LSTM for temporal dependencies, Layer Normalization, Dropout for regularization, and Global Average Pooling before a Dense output layer. The model was trained with EarlyStopping and ReduceLROnPlateau callbacks. Evaluation was performed using MAE, RMSE, and R² Score. Finally, a recursive 30-day forecast was generated using the last available window of features.

## Main Results
The hybrid CNN-BiLSTM model demonstrated strong performance on the test set, capturing the general trend and turning points in NVIDIA's stock price. The model was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score. Among the 19 engineered features, RSI, short-term returns (1-day and 3-day), and moving average ratios were the most influential for prediction accuracy. The 30-day forecast projects a price trajectory based on learned patterns, displayed alongside historical data in the Streamlit dashboard with a bullish or bearish signal badge. The interactive app also presents KPI cards for the latest close price, 30-day change, 52-week range, and average trading volume.

## Prescriptive Insights
Time series forecasting of individual stocks is inherently limited by market volatility and external factors not captured in historical price data alone. The model could be improved by incorporating additional features such as trading sentiment, macroeconomic indicators, or earnings calendar data. Ensemble approaches combining LSTM with transformer-based architectures may further improve prediction accuracy. This type of analysis serves as a valuable educational tool for understanding how deep learning can be applied to financial time series, but predictions should not be used as the sole basis for investment decisions.

## Authors
•	Laura Victoria Miquel Herrera
• Karla Sofía Cantú Zendejas
• Diego Sánchez Magaña
• Saúl Josué Ruiz González
• Rodrigo Rivas González

## Technologies & Tools
•	Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Plotly)

•	Streamlit (interactive dashboard)

•	Jupyter Notebook

•	GitHub repository
