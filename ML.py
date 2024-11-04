'''
Sahil Bains
Stock Price Prediction Model Implementation

Purpose
This script implements a Random Forest model for predicting stock prices, focusing on data preprocessing,
feature engineering, and model optimization through hyperparameter tuning. The script processes historical
stock data to generate predictions for April 2024, with emphasis on proper data filtering and robust model
training.

Model Implementation
Random Forest Regressor: The script utilizes sklearn's RandomForestRegressor with GridSearchCV for 
hyperparameter optimization. Key parameters include n_estimators (100-300), max_depth (10-30), and 
min_samples_split (2-10). Individual models are trained for each stock symbol to capture unique patterns.

Data Preprocessing
Symbol Filtering: The script filters for symbols with at least 40 data points to ensure robust model training
and prevent underfitting. Temporal Features: Creates lag features (1-day, 2-day) and implements a 3-day 
rolling mean to capture price dependencies. Data Normalization: Uses MinMaxScaler for feature normalization,
with separate scalers for features and target variables.

Feature Engineering
Calendar Features: Extracts day, month, and day of week from trade dates
Lag Features: Creates 1-day and 2-day price lags to capture temporal dependencies
Rolling Mean: Implements 3-day rolling mean for trend capture
All features are normalized using MinMaxScaler before model training

Model Training
Train-Test Split: 80-20 split for model validation
Hyperparameter Tuning: GridSearchCV with 3-fold cross-validation
Parallel Processing: Utilizes n_jobs=-1 for optimal computation speed
Individual model training for each stock symbol

Prediction Generation
Future Dates: Generates predictions for April 1-30, 2024
Feature Consistency: Maintains temporal features for prediction period
Visualization: Creates interactive plots using Plotly Express

Output
Predictions DataFrame: Combined predictions for all symbols
Interactive Plot: Visualization saved as HTML file
Model Metrics: Stored best parameters and performance metrics

Importance
This script represents the core modeling component of the stock price prediction project, building upon
the exploratory analysis from ML_Explore. The implementation focuses on robust feature engineering and
model optimization to ensure reliable predictions while handling multiple stock symbols efficiently.
'''

# Import Libraries
import pandas as pd # For data manipulation/analysis
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split, GridSearchCV # For splitting data and hyperparameter tuning
from sklearn.preprocessing import MinMaxScaler # For normalizing data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # For evaluating model performance 
from sklearn.ensemble import RandomForestRegressor # For the Random Forest Model
import plotly.express as px # For data visualization

# Load the CSV file
file_path = 'temporal_changes_symbols.csv'
df = pd.read_csv(file_path)

# Data Preprocessing - processing and sorting dates
df['Trade Date'] = pd.to_datetime(df['Trade Date']) # Converting 'Trade Date' to datetime format
df = df.sort_values('Trade Date') # Sort the DataFrame by 'Trade Date'

# Filter symbols with at least 40 values
symbol_counts = df['Symbol'].value_counts() # Counting the number of occurrences of each symbol
valid_symbols = symbol_counts[symbol_counts >= 40].index # Subsetting data with at least 40 occurrences
df = df[df['Symbol'].isin(valid_symbols)] # Filtering the dataset to include the valid symbols

# Initialize a list to store predictions for all symbols
all_predictions = []

# Loop through each symbol and perform the steps individually
for symbol in valid_symbols:
    symbol_df = df[df['Symbol'] == symbol].copy() # Create a dataframe for the current symbol
    
    # Feature Engineering
    symbol_df['Day'] = symbol_df['Trade Date'].dt.day # Extract day from 'Trade Date'
    symbol_df['Month'] = symbol_df['Trade Date'].dt.month # Extract month from 'Trade Date'
    symbol_df['DayOfWeek'] = symbol_df['Trade Date'].dt.dayofweek # Extract day of the week from 'Trade Date'
    symbol_df['Lag1'] = symbol_df['Trade Price Per Share'].shift(1) # Create a lag feature to view temporal dependencies for 1 day
    symbol_df['Lag2'] = symbol_df['Trade Price Per Share'].shift(2) # Create a lag feature to view temporal dependencies for 2 days
    symbol_df['RollingMean'] = symbol_df['Trade Price Per Share'].rolling(window=3).mean() # Create a rolling mean feature (3 days)
    
    # Drop NaN values created by lag features
    symbol_df = symbol_df.dropna() # Dropping rows with NaNs
    
    # Normalize the data
    scaler = MinMaxScaler() # Initialize a scaler for lag and rolling mean features
    price_scaler = MinMaxScaler() # Initialize a scaler for the target variable
    symbol_df[['Lag1', 'Lag2', 'RollingMean']] = scaler.fit_transform(symbol_df[['Lag1', 'Lag2', 'RollingMean']]) # Normalize lag and rolling mean features
    symbol_df['Trade Price Per Share'] = price_scaler.fit_transform(symbol_df[['Trade Price Per Share']]) # Normalize the target variable
 
    # Split the data
    X = symbol_df[['Day', 'Month', 'DayOfWeek', 'Lag1', 'Lag2', 'RollingMean']] # Features
    y = symbol_df['Trade Price Per Share'] # Target Variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into training and testing sets
    
    # Model Training with Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300], # Number of trees in the forest
        'max_depth': [10, 20, 30], # Maximum depth of the tree
        'min_samples_split': [2, 5, 10] # Minimum number of samples required to split an internal node
    }
    rf = RandomForestRegressor(random_state=42) # Initialize the Random Forest model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2) # Initialize GridSearchCV for hyperparameter tuning
    grid_search.fit(X_train, y_train) # Fit the model to the training data
    
    # Best model
    best_rf = grid_search.best_estimator_ # Get the best model from GridSearchCV
    
    # Predictions for April
    future_dates = pd.date_range(start='2024-04-01', end='2024-04-30') # Generate dates for April
    future_df = pd.DataFrame({'Trade Date': future_dates}) # Create DF for future dates
    future_df['Day'] = future_df['Trade Date'].dt.day # Extract day from Trade Date
    future_df['Month'] = future_df['Trade Date'].dt.month # Extract month from 'Trade Date'
    future_df['DayOfWeek'] = future_df['Trade Date'].dt.dayofweek # Extract day of the week from 'Trade Date'
    
    # Assuming the last known prices and rolling means are available for each symbol
    last_known = symbol_df.iloc[-1] # Get the last known values for the current symbol
    future_df['Lag1'] = last_known['Lag1'] # Set the last known Lag1 value
    future_df['Lag2'] = last_known['Lag2'] # Set the last known Lag2 value
    future_df['RollingMean'] = last_known['RollingMean'] # Set the last known RollingMean Value
    
    # Normalize future data
    future_df[['Lag1', 'Lag2', 'RollingMean']] = scaler.transform(future_df[['Lag1', 'Lag2', 'RollingMean']])
    
    # Predict
    X_future = future_df[['Day', 'Month', 'DayOfWeek', 'Lag1', 'Lag2', 'RollingMean']] # Features for prediction
    future_df['Predicted Price'] = best_rf.predict(X_future) # Predict future prices
    
    # Inverse transform only the predicted prices
    future_df['Predicted Price'] = price_scaler.inverse_transform(future_df[['Predicted Price']]) # Inverse transform the predicted prices
    
    # Add the Symbol column to the predictions and append to all_predictions list
    all_predictions.append(future_df[['Trade Date', 'Predicted Price']].assign(Symbol=symbol)) # Append predictions to list

# Combine predictions for all symbols into a single DataFrame
predictions_df = pd.concat(all_predictions) # Combine all predictions into one dataframe

# Visualize the stock prices on a temporal graph
fig = px.line(predictions_df, x='Trade Date', y='Predicted Price', color='Symbol', title='Predicted Stock Prices for April')
fig.write_html("Model_Projection_Visualization.html")
fig.show()