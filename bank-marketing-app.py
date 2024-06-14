import streamlit as st
import pandas as pd
import joblib


# Define the input form
def main():
    st.title("Bank Marketing Prediction")

    # Input fields
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = st.selectbox('Default', ['no', 'yes', 'unknown'])
    housing = st.selectbox('Housing Loan', ['no', 'yes', 'unknown'])
    loan = st.selectbox('Personal Loan', ['no', 'yes', 'unknown'])
    contact = st.selectbox('Contact', ['cellular', 'telephone'])
    month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.number_input('Duration', min_value=0, value=0)
    campaign = st.number_input('Campaign', min_value=0, value=0)
    pdays = st.number_input('Pdays', min_value=-1, value=-1)
    previous = st.number_input('Previous', min_value=0, value=0)
    poutcome = st.selectbox('Poutcome', ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.number_input('Emp Var Rate', value=0.0)
    cons_price_idx = st.number_input('Cons Price Index', value=0.0)
    cons_conf_idx = st.number_input('Cons Confidence Index', value=0.0)
    euribor3m = st.number_input('Euribor 3 Month', value=0.0)
    nr_employed = st.number_input('Nr Employed', value=0.0)

    # Collect input data into a DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed]
    })

    # Perform prediction
    if st.button('Predict'):
        result = bank_marketing_prediction(input_data)
        st.write('Prediction:', 'Yes' if result == 1 else 'No')

# Prediction function
def bank_marketing_prediction(input_data):
    # Load the model
    with open('best_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)    # Ensure the input data is in DataFrame format with the correct columns
    input_data_df = pd.DataFrame(input_data)
    prediction = loaded_model.predict(input_data_df)
    return prediction[0]

if __name__ == '__main__':
    main()
