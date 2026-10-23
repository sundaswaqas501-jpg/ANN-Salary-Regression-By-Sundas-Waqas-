



import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

st.set_page_config(
    page_title="ANN Salary Prediction",
    page_icon="💰",
    layout="centered"
)

@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model("salary_model.keras")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder

model, scaler, encoder = load_model_files()

st.title("💰 ANN Salary Prediction")
st.write("Predict Estimated Salary using an Artificial Neural Network.")

st.warning("This project is for educational purposes only.")

with st.form("salary_form"):
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    products = st.number_input("Number of Products", 1, 4, 2)
    credit_card = st.selectbox("Has Credit Card", [1, 0])
    active_member = st.selectbox("Active Member", [1, 0])
    exited = st.selectbox("Exited", [0, 1])

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": credit_card,
        "IsActiveMember": active_member,
        "Exited": exited
    }])

    categorical_cols = ["Geography", "Gender"]
    numerical_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"
    ]

    encoded_data = encoder.transform(input_data[categorical_cols])
    scaled_data = scaler.transform(input_data[numerical_cols])

    final_data = np.hstack([scaled_data, encoded_data]).astype("float32")

    prediction = model.predict(final_data, verbose=0)[0][0]

    st.success(f"Predicted Estimated Salary: {prediction:,.2f}")



st.divider()

st.header("📚 Reflection Questions & Answers")

with st.expander("Q1. Why did we remove sigmoid from the output layer?"):
    st.write(
        "We removed sigmoid because it restricts the output "
        "between 0 and 1, while salary is a continuous numerical "
        "value. Therefore, we used linear activation."
    )

with st.expander("Q2. Why is Binary Cross-Entropy not suitable here?"):
    st.write(
        "Binary Cross-Entropy is designed for classification "
        "tasks, whereas salary prediction is a regression task. "
        "Therefore, we used Mean Squared Error (MSE) as the loss function."
    )

with st.expander("Q3. What does MAE mean in salary units?"):
    st.write(
        "MAE represents the average absolute difference between "
        "predicted and actual salaries, measured in the same "
        "units as the salary data."
    )

with st.expander("Q4. If RMSE is much larger than MAE, what might that suggest?"):
    st.write(
        "It may indicate that some predictions have large errors "
        "or that outliers are affecting the model. RMSE penalizes "
        "large errors more strongly than MAE."
    )

with st.expander("Q5. What are the limitations of predicting salary from this classroom dataset?"):
    st.write(
        "This dataset is designed for classroom practice and may "
        "not represent real-world salary patterns. It may lack "
        "important salary-related features and contain biases. "
        "Therefore, this model should not be used for real banking "
        "or other important financial decisions."
    )


    

 

