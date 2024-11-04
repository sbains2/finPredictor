# Stock Price Prediction Using Machine Learning

## Personal Journey & Motivation
Coming from a data science background with extensive experience in data wrangling and implementing machine learning models, my summer internship at Badgley Phelps, a wealth management firm, presented an exciting opportunity to bridge data science with finance. The firm had specialized teams across Research, Operations, Financial Planning, and Trading, each playing a crucial role in client service. While my primary focus was on client relationships, income projections, and estate planning, I identified an opportunity to apply technical expertise in a field traditionally driven by fundamental analysis.

This experience deepened my understanding of how data-driven decisions can enhance portfolio management. At Badgley Phelps, data analysis was primarily performed using Excel, which, while functional, lacked the sophistication of advanced data science tools. Recognizing this gap, I proposed using object-oriented programming to sample and analyze trading data, aiming to create a system for identifying high-return stocks—a project that ultimately bridged finance and data science.

## Project Overview
Leveraging machine learning to predict stock prices, this project uses a Random Forest model with engineered features to improve predictive accuracy, outperforming traditional methods. While the data used for analysis is proprietary and confidential to Badgley Phelps and cannot be attached to this project, the approach demonstrates how historical trading data can be effectively transformed into actionable insights for portfolio management. The project further explores the effectiveness of advanced visualization and feature engineering in capturing complex market patterns.

## Purpose and Real-World Application
The purpose of this project was to introduce data science applications to Badgley Phelps, showcasing how machine learning can optimize manual research typically conducted via Bloomberg, Yahoo Finance, Morningstar, and JPM. This was an ideal opportunity to combine data science and business for impactful results. By optimizing the research process, we could enhance client interactions by reallocating time for more meaningful discussions, fostering trust, and better supporting clients’ financial needs. As data science becomes more ubiquitous across industries, I believe it will revolutionize workflows, ultimately allowing professionals to dedicate more time to serving people and improving their lives.

## Project Evolution
Initially focused on analyzing trade metrics like volume and activity, the project evolved as I became interested in the potential of combining historical trade volumes with price movements to forecast future trends. This journey led to the development of a comprehensive approach that leverages feature engineering to improve predictive accuracy.

## Technical Objective
The primary objective was to develop a machine learning model capable of predicting stock prices for the upcoming month using historical trading data. I compared a Random Forest model against baseline approaches, including Linear Regression and a naive method that used the previous day's price.

## Project Components

### Script Overview
- **ML_Explore.py**: Initial exploratory data analysis script focused on data preprocessing and visualization. Handles the cleaning of raw trading data, implements temporal feature creation, and generates interactive visualizations to understand price patterns across different symbols. Outputs processed data for subsequent modeling.
- **ML.py**: Core implementation of the Random Forest model for stock price prediction. Focuses on feature engineering, model training pipeline development, and initial hyperparameter tuning. Creates lag features, rolling means, and calendar-based features while implementing parallel processing for efficient multi-symbol handling.
- **ML_Optimize.py**: Advanced model optimization script that introduces baseline comparisons against Linear Regression and naive prediction approaches. Demonstrates the Random Forest model's superior performance (98% R² score) compared to simpler methods. Implements extensive hyperparameter tuning and generates comprehensive performance metrics for model evaluation.

### Visualization Overview
- **temporal_changes_symbols_plot.html**:
Temporal representation of stock price changes from January 2024 - March 2024
[View Stock Price Changes](/temporal_changes_symbols_plot.html)

- **Model_Projection_Visualization.html**:
Random Forest prediction of Stock price changes
[View Stock Price Predictions](/Model_Projection_Visualization.html)


## Dataset
- **Time Period**: January to March 2024
- **Key Features**: Stock symbols, trade dates, and trade prices

### Data Preprocessing
- Removal of dollar signs
- Conversion of prices to numeric values
- Date formatting standardization

## Methodology

### 1. Data Preprocessing
- **Loading Data**: Implemented robust data loading from CSV files with over 45,000 observations.
- **Cleaning Operations**:
  - Removed currency symbols
  - Converted price strings to numeric values
  - Standardized date formats
- **Data Filtering**: Applied date range filtering (Jan 2, 2024 - Mar 28, 2024)

### 2. Feature Engineering
- **Temporal Features**:
  - Created 1-day and 2-day lag features
  - Implemented rolling mean calculations
- **Date-Based Features**:
  - Extracted day, month, and day of the week
  - Created cyclical features for capturing temporal patterns

Using temporal features allowed for a better understanding of stock price trends and improved model accuracy. This enhancement reduced errors significantly (lower MSE and MAE) compared to earlier versions, highlighting the importance of time-based patterns in stock price prediction.

### 3. Model Development
- **Random Forest Implementation**:
  - Hyperparameter tuning via GridSearchCV
  - Selected optimal tree depth and number of estimators
- **Baseline Models**:
  - Linear Regression
  - Naive prediction using previous day's price
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score

### 4. Visualization Pipeline
- **Interactive Visualizations**: Implemented using Plotly.
- **Temporal Analysis**: Created dynamic visualizations of price changes over time.
- **Prediction Visualization**: Generated forecast visualizations for April.

## Results & Analysis
The Random Forest model demonstrated superior performance compared to baseline models:
- **Random Forest Performance**: 98% R² score.
- **Comparison to Baselines**:
  - 76% reduction in MSE compared to Linear Regression.
  - 90% reduction in MSE compared to Naive Predictor.
- **Additional Benefits**:
  - Improved handling of market volatility.
  - Enhanced insights into feature importance.

## Key Learnings
- **Feature Engineering Impact**: Lag features and rolling means significantly boosted model performance.
- **Hyperparameter Optimization**: GridSearchCV was crucial for fine-tuning.
- **Model Comparison**: The ensemble approach of Random Forest outperformed simpler models.

## Technical Challenges & Solutions
- **Data Quality**: Implemented robust pipelines to handle missing values and outliers.
- **Model Optimization**: Used cross-validation and hyperparameter tuning to prevent overfitting.
- **Scalability**: Optimized code to efficiently handle multiple stock symbols.

## Future Enhancements
- **Advanced Models**: Integrate LSTM networks for improved temporal modeling. LSTM’s sequential nature could capture time-dependent patterns more effectively, potentially leading to more accurate stock price predictions.
- **Sentiment Analysis**: Including sentiment analysis on financial news could add valuable context to the feature engineering process, enhancing model accuracy by accounting for market sentiment.
- **Data Expansion**: Incorporate larger historical datasets.
- **Feature Enhancement**: Add technical indicators and market sentiment metrics.

## Conclusion
This project successfully demonstrated the potential of machine learning in stock price prediction, particularly through the effective use of Random Forest models. The comprehensive approach to feature engineering and model evaluation provides a strong foundation for further development in algorithmic trading strategies. By bridging the gap between finance and data science, this project offers a novel perspective on portfolio optimization that the firm can potentially adopt in the future.

During my internship, I was able to share the details of my machine learning-powered stock price prediction approach with the trading team at Badgley Phelps. By walking through the data wrangling, model training, and visualization techniques, I presented an alternative method for the firm to conduct market analysis compared to their traditional fundamental analysis. This data-driven approach was well-received, as it showcased the power of leveraging historical trading data and advanced analytics to enhance investment strategies.

## Technologies Used
- **Python Libraries**: `pandas`, `numpy`, `scikit-learn`, `plotly`
- **Machine Learning**: `RandomForestRegressor`, `LinearRegression`
- **Data Visualization**: `Plotly`, `Seaborn`
- **Data Processing**: `pandas`, `numpy`
