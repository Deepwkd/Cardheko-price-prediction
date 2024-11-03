Car Price Prediction Project
This repository contains a Car Price Prediction project, including the model development, preprocessing, exploratory data analysis (EDA), and a Streamlit application for predicting car prices based on user input. The project uses machine learning and regression analysis to predict the price of a used car based on its features.

Project Files
Car_Price_Prediction_app.py:

A Streamlit-based application that allows users to input car details and get a predicted car price.
Loads a pre-trained model pipeline and uses it to predict prices based on various car attributes.

Key features:
Encodes categorical variables with OneHotEncoding.
Standardizes numerical features.
Uses a trained GradientBoostingRegressor model.
The app also displays the dataset and predicted price, formatted in an easy-to-read way.

Eda_ML.ipynb:

A Jupyter Notebook that conducts Exploratory Data Analysis (EDA) on the car dataset.
Provides insights into data distribution, feature importance, and relationships between variables.

Key analyses:
Distribution of each feature, including numerical and categorical variables.
Correlation analysis and insights to guide feature selection.
Creates visualizations to assist with understanding key patterns within the dataset.

Preprocessing.ipynb:

A Jupyter Notebook focused on data preprocessing steps.
Processes data for model training, including:
Handling missing values and outliers.
Encoding categorical variables with OneHotEncoder and scaling numerical features with StandardScaler.
Defines a pipeline for transforming both categorical and numerical features, preparing the data for regression modeling.

Model Information
Model Type: Gradient Boosting Regressor

Preprocessing:
OneHotEncoder: Used for categorical columns to ensure the model can interpret these as binary features.
StandardScaler: Used for numerical columns to standardize features and improve model performance.
Training and Evaluation: The model was tuned and evaluated with appropriate metrics such as mean absolute error (MAE) and R-squared.


How to Run the Project

Clone the repository:
git clone https://github.com/yourusername/car_price_prediction.git
cd car_price_prediction

Install required packages:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run Car_Price_Prediction_app.py

Requirements
Python 3.7 or higher
Streamlit
scikit-learn
pandas
numpy
joblib
