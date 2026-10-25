
# ANN Salary Regression

## 1. Project Overview
This project uses an Artificial Neural Network (ANN) to predict EstimatedSalary as a continuous numerical value. It converts a churn classification workflow into a regression workflow.

This project is for educational purposes only.

## 2. Dataset
The project uses the Churn_Modelling.csv dataset.

The target variable is EstimatedSalary. Identifier columns such as RowNumber, CustomerId, and Surname are removed from the input features.

## 3. Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow and Keras
- Matplotlib and Seaborn
- Streamlit

## 4. Data Preprocessing
- Inspected the dataset and checked missing values and duplicates.
- Removed identifier columns.
- Separated input features (X) and target (y).
- Split the dataset into training and testing sets.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features.

Preprocessing objects were saved for reuse in the Streamlit application.

## 5. ANN Architecture
The Artificial Neural Network contains:
- Input layer based on the processed features
- Dense layer with 64 neurons and ReLU activation
- Dense layer with 32 neurons and ReLU activation
- Output layer with one neuron and linear activation

The model uses the Adam optimizer and Mean Squared Error (MSE) loss.

## 6. Model Evaluation
The model is evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Actual evaluation results are available in the assignment notebook.

## 7. How to Run
1. Install the required libraries:
   pip install -r requirements.txt

2. Open the notebook and run all cells from top to bottom.

3. Run the Streamlit application:
   streamlit run app.py

## 8. Deployment
The application is deployed using Streamlit Community Cloud.

Live App:
https://annsalaryregressionsundaswaq.streamlit.app/

GitHub Repository:
https://github.com/sundaswaqas501-jpg/ANN-Salary-Regression-By-Sundas-Waqas-⁠

## 9. Classification vs Regression
Classification predicts a category or class, such as whether a customer will leave or stay. Regression predicts a continuous numerical value, such as EstimatedSalary.

This project uses a linear output layer, MSE loss, and MAE, RMSE, and R² metrics instead of classification metrics.

## 10. Limitations
This is a classroom learning project. The dataset may not represent real-world salary patterns, and the available features may not reliably determine actual salaries.

The model should not be used for real banking, hiring, or other important financial decisions.
