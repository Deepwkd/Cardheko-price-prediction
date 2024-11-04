import streamlit as st
import pandas as pd
import joblib
import pickle

# Loading pre-trained model
model= joblib.load('car_price_model_pipeline.pkl')

# dataset to populate options
data = pd.read_excel("Final_Cardheko_list.xlsx")

# Converting all columns that require encoding to strings to match training format
data['Car_model'] = data['Car_model'].astype(str)
data['Fuel_type'] = data['Fuel_type'].astype(str)
data['Body_type'] = data['Body_type'].astype(str)
data['city'] = data['city'].astype(str)

# Function for predicting car price
def predict_price(car_data):
    X = pd.DataFrame(car_data, index=[0])
    prediction = model.predict(X)
    return prediction[0]

# Streamlit app
def main():
    st.title('Car Price Prediction')
    
    # Option to show dataset for reference
    if st.checkbox('Show Dataset'):
        st.write(data)

    # Sidebar for user inputs
    st.sidebar.header('Enter Car Details')

    # Car model input
    car_model = st.sidebar.selectbox('Car Model', data['Car_model'].unique())

    # Other inputs
    year_of_manufacture = st.sidebar.number_input('Year of Manufacture', min_value=1900, max_value=2024, value=2000)
    kilometers_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, value=0)
    num_previous_owners = st.sidebar.number_input('Number of Previous Owners', min_value=0, value=0)

    # Transmission, fuel, body, and city inputs
    transmission_type = st.sidebar.selectbox('Transmission Type', ['Manual', 'Automatic'])
    fuel_type = st.sidebar.selectbox('Fuel Type', data['Fuel_type'].unique())
    body_type = st.sidebar.selectbox('Body Type', data['Body_type'].unique())
    city = st.sidebar.selectbox('City', data['city'].unique())

    # Predicting button
    if st.sidebar.button('Predict'):
        # user inputs ensuring all categories are strings
        user_inputs = {
            'Car_model': str(car_model),
            'Year_of_car_manufacture': year_of_manufacture,
            'Kilometers_driven': kilometers_driven,
            'Number_of_previous_owners': num_previous_owners,
            'Transmission_type': str(transmission_type),
            'Fuel_type': str(fuel_type),
            'Body_type': str(body_type),
            'city': str(city)
        }
        
        price_prediction = predict_price(user_inputs)
        st.sidebar.subheader(f'Predicted Price: {price_prediction:,.2f}')

if __name__ == '__main__':
    main()
