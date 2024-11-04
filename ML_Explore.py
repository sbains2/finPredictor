'''
Sahil Bains
Exploratory Data Analysis and Visualization for Stock Price Prediction

Purpose
This script is the first step in the stock price prediction project. It focuses on the initial data preprocessing, 
exploratory data analysis, and visualization of the temporal changes in stock prices over a specific time period. 
This exploration lays the groundwork for the subsequent feature engineering and model development stages.

Data Preprocessing
Load CSV File: The script loads the stock data from a CSV file. The dtype parameter is used to ensure the 'Symbol' 
column is read as a string. Remove Dollar Signs and Convert to Numeric: The 'Trade Price Per Share' column is cleaned
by removing the dollar sign and comma using a regular expression, and then converting the values to float.
Format Trade Date: The 'Trade Date' column is converted to datetime format for better handling of temporal data.
Filter Data Range: The script filters the data to include only the time period from January 2nd, 2024 to March 28th,
2024. Group Data: The data is grouped by 'Symbol' and 'Trade Date', and the mean 'Trade Price Per Share' is calculated
for each day.

Visualization
Temporal Histogram: An interactive line plot is created using Plotly Express to visualize the temporal changes in stock 
prices for all unique symbols. This plot is saved as an HTML file named 'temporal_changes_symbols_plot.html'.
PDF Report: A PDF document is generated using the FPDF library, which includes a table with the symbol, associated trade
price, and trade date for each data point.

Output
CSV File: The grouped data is saved to a CSV file named 'temporal_changes_symbols.csv' in the current working directory.
HTML File: The interactive temporal histogram is saved as an HTML file named 'temporal_changes_symbols_plot.html'.
PDF Report: A PDF document is generated, showcasing the temporal changes in stock prices.

Importance and Next Steps
This exploratory data analysis and visualization script serves as the foundation for the stock price prediction project.
It provides valuable insights into the temporal patterns of stock price movements, which will inform the feature engineering
process. By identifying stocks with sufficient data, this script also helps to ensure the machine learning model is not 
underfitted due to lack of training data. The next step would be to leverage the insights gained from this exploration to
develop more sophisticated feature engineering techniques and build the predictive machine learning models.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import plotly.express as px

# Data Preprocessing
## Load the CSV file
file_path = 'C:\\Users\\sahilb\\Desktop\\Software\\temporal_changes_symbols.csv'
df = pd.read_csv(file_path, dtype={'Symbol': str})  # Ensure 'Symbol' is read as a string

## Remove dollar signs and convert 'Trade Price Per Share' to numeric
df['Trade Price Per Share'] = df['Trade Price Per Share'].replace('[\\$,]', '', regex=True).astype(float)

## Formatting Trade Date
df['Trade Date'] = pd.to_datetime(df['Trade Date'])

# Filter data from January 2nd, 2024 to March 28th, 2024
start_date = '2024-01-02'
end_date = '2024-03-28'
mask = (df['Trade Date'] >= start_date) & (df['Trade Date'] <= end_date)
filtered_df = df.loc[mask]

# Group by 'Symbol' and 'Trade Date' and calculate the mean 'Trade Price Per Share' for each day
grouped_df = filtered_df.groupby(['Symbol', 'Trade Date'])['Trade Price Per Share'].mean().reset_index()

# Plotting the temporal histogram for all unique 'Symbols' in one plot
fig = px.line(grouped_df, x='Trade Date', y='Trade Price Per Share', color='Symbol', markers=True,
              title='Temporal Changes in Stock Prices (Jan-Mar 2024)')
fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
fig.write_html('temporal_changes_symbols_plot.html')

print("Interactive temporal histogram for all symbols created successfully.")

# Create a PDF document
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add a new page to the PDF
pdf.add_page()

# Set font for the PDF
pdf.set_font("Arial", size=12)

# Add title for the current plot
pdf.cell(200, 10, txt='Temporal Changes in Stock Prices (Jan-Mar 2024)', ln=True, align='C')

# Add table headers
pdf.cell(60, 10, txt='Symbol', border=1)
pdf.cell(60, 10, txt='Associated Value', border=1)
pdf.cell(60, 10, txt='Date', border=1)
pdf.ln()

# Add table rows
for index, row in grouped_df.iterrows():
    pdf.cell(60, 10, txt=row['Symbol'], border=1)
    pdf.cell(60, 10, txt=str(row['Trade Price Per Share']), border=1)
    pdf.cell(60, 10, txt=row['Trade Date'].strftime('%Y-%m-%d'), border=1)
    pdf.ln()

# Save the grouped data to a CSV file
output_csv_path = 'temporal_changes_symbols.csv'
grouped_df.to_csv(output_csv_path, index=False)

print(f"CSV with temporal changes in stock prices created successfully at {output_csv_path}.")
