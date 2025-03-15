# Pittsburgh Housing Prices Modeling Project

## Overview

This project analyzes and models housing prices in Pittsburgh, with a focus on identifying the most and least expensive neighborhoods based on various factors such as property characteristics, neighborhood grades, and the year of sale. The project uses statistical models like linear regression and weighted least squares, as well as diagnostics to explore housing price trends across Pittsburgh's ZIP codes.

### Key Objectives:
1. **Data Preprocessing**: Clean and prepare the housing data for analysis.
2. **Exploratory Data Analysis (EDA)**: Explore housing price distributions, examine relationships between variables, and visualize key insights.
3. **Statistical Modeling**: Build regression models to predict housing prices based on different features, and evaluate model performance.
4. **Model Diagnostics**: Check for assumptions like multicollinearity, heteroskedasticity, and influential points.
5. **Final Analysis**: Identify the most and least expensive neighborhoods using adjusted means and visualize trends over time.

## Libraries Used

This project leverages the following R libraries:

- **tidyverse**: For data manipulation and visualization.
- **car**: For regression diagnostics and VIF (Variance Inflation Factor).
- **ggplot2**: For creating visualizations.
- **MASS**: For statistical modeling functions.
- **readxl**: For reading Excel files (if needed).
- **softImpute**: For matrix factorization techniques (if any missing values).
- **stargazer**: For presenting summary statistics and regression results.
- **gridExtra**: For arranging multiple plots in a grid.

## Data

The dataset used in this project (`pgh_dropped.csv`) contains housing data for Pittsburgh. The key columns include:

- **PRICE**: The sale price of the house.
- **PROPERTYZIP.x**: The ZIP code of the property.
- **GRADEDESC**: The neighborhood rating (e.g., "EXCELLENT," "AVERAGE").
- **YEARBLT**: The year the house was built.
- **SALEYEAR**: The year the house was sold.
- **FINISHEDLIVINGAREA**: The finished living area of the house.
- **LOTAREA**: The lot size.
- **FULLBATHS**: The number of full bathrooms.
- **HALFBATHS**: The number of half bathrooms.
- **STORIES**: The number of stories in the house.
- **CONDITION**: The condition of the house (e.g., "EXCELLENT," "GOOD").

### Data Cleaning & Preprocessing
- **Missing Values**: Rows with missing values in important columns like `GRADEDESC` are removed.
- **Factor Conversion**: The `GRADEDESC` column (neighborhood grade) is converted into an ordered factor for statistical analysis.
- **Log Transformation**: Some continuous variables like `PRICE`, `LOTAREA`, and `FINISHEDLIVINGAREA` are log-transformed to stabilize variance and make the distribution more normal.

## Statistical Models and Analysis

### 1. Exploratory Data Analysis (EDA)
The initial steps involve exploring the data with summary statistics and visualizations to identify patterns and understand the structure of the data:
- **Summary Statistics**: Basic descriptive statistics for the dataset are calculated.
- **Price by ZIP Code**: The average housing prices per ZIP code are computed and visualized using a bar chart with error bars indicating the standard error of the mean.

### 2. Linear Regression Models
A series of linear regression models are constructed to predict housing prices based on several independent variables.

#### a. Full Model (Max Model)
The **max model** includes all potential predictors of house price, such as `LOTAREA`, `STYLE`, `YEARBLT`, `GRADEDESC`, `CONDITION`, and more.

#### b. Null Model (Min Model)
The **null model** includes only an intercept, essentially predicting the mean price without any independent variables.

#### c. Stepwise Model Selection
The **best model** is selected using a stepwise selection process. This function selects the best predictors based on AIC (Akaike Information Criterion), which balances model fit and complexity.

### 3. Model Diagnostics
- **Multicollinearity**: The variance inflation factor (VIF) is calculated to check for multicollinearity among the predictors. A VIF value greater than 10 indicates high collinearity.
- **Influential Points**: Cook's Distance is used to identify and remove highly influential points that might distort the model’s estimates.
- **Heteroskedasticity**: The Breusch-Pagan test is used to detect heteroskedasticity (non-constant variance of residuals). If detected, robust standard errors are applied to correct for this issue.

### 4. Weighted Least Squares (WLS)
When heteroskedasticity is present, **Weighted Least Squares (WLS)** regression is used, where the weights are inversely proportional to the fitted values from the initial model.

### 5. Box-Cox Transformation
The **Box-Cox transformation** is applied to the dependent variable (`PRICE`) to identify an optimal transformation (lambda) that improves the model’s normality.

### 6. Analysis of Covariance (ANCOVA)
ANCOVA is performed to compare the mean prices of homes across different ZIP codes, adjusting for other variables in the model.

### 7. Adjusted Means
Finally, **adjusted means** for the housing prices across ZIP codes are calculated, accounting for nuisance variables like `GRADEDESC`, `SALEDESC.x`, and `HEATINGCOOLING`. These adjusted means are visualized to identify which ZIP codes are the most and least expensive.

## Visualizations
Various visualizations are generated to help understand the trends in the data:
- **Average Prices by ZIP Code**: A bar plot with error bars for each ZIP code showing the average price and its confidence interval.
- **Adjusted Prices by ZIP Code**: A bar plot with adjusted prices based on the model’s results, including error bars and confidence intervals.
- **Price Trends Over Time**: A line plot showing the adjusted prices for the top and bottom ZIP codes over time.

## Usage

To use the project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/pittsburgh-housing-prices.git
