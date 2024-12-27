import os
import pickle 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import warnings 
warnings.filterwarnings('ignore')

# Title and Description
st.title("Fitness Tracker System")
st.write("This app allows users to predict fitness metrics such as Calories Burned based on input features.")

# Create synthetic training data
X_train, y_train = make_regression(n_samples=100, n_features=11, noise=0.1)

# Initialize and fit the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the fitted model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the pre-trained model
def load_model():
    try:
        with open(r"C:\Users\ASUS\OneDrive\Desktop\Major_Project (Scifor Technologies)\decision_tree_model.pkl", 'rb') as file:
            model = pickle.load(file)
        return model
    except EOFError as e:
        st.error("Error loading model: EOFError - Ran out of input")
        return None
    except FileNotFoundError as e:
        st.error("Error: Model file not found.")
        return None
    except Exception as e:
        st.error("An error occurred while loading the model:")
        st.error(str(e))
        return None

model = load_model()

# Check if the model was loaded successfully
if model is None:
    st.stop()  # Stop the app if the model is not loaded

# Input form for user data
st.header("Enter Your Fitness Details:")
age = st.slider("Age", 10, 80, 25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0, step=0.1)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
max_bpm = st.number_input("Max BPM", min_value=100, max_value=200, value=150)
avg_bpm = st.number_input("Average BPM", min_value=60, max_value=180, value=120)
resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=70)
session_duration = st.number_input("Session Duration (hours)", min_value=0.5, max_value=5.0, value=1.0)
fat_percentage = st.number_input("Fat Percentage (%)", min_value=5.0, max_value=50.0, value=20.0)
water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=5.0, value=2.0)
workout_frequency = st.slider("Workout Frequency (days/week)", 1, 7, 3)

# Preprocess inputs
def preprocess_input(age, weight, height, max_bpm, avg_bpm, resting_bpm, session_duration,
                     fat_percentage, water_intake, workout_frequency):
    bmi = weight / (height ** 2)
    data = pd.DataFrame({
        'Age': [age],
        'Weight (kg)': [weight],
        'Height (m)': [height],
        'Max_BPM': [max_bpm],
        'Avg_BPM': [avg_bpm],
        'Resting_BPM': [resting_bpm],
        'Session_Duration (hours)': [session_duration],
        'Fat_Percentage': [fat_percentage],
        'Water_Intake (liters)': [water_intake],
        'Workout_Frequency (days/week)': [workout_frequency],
 'BMI': [bmi]
    })
    # Scale inputs
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Button for Prediction
if st.button("Predict Calories Burned"):
    # Preprocess the input data
    input_data = preprocess_input(age, weight, height, max_bpm, avg_bpm, resting_bpm,
                                  session_duration, fat_percentage, water_intake, workout_frequency)
    # Predict using the model
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")

# Additional Section: Insights
st.header("Insights and Recommendations:")
st.write("""
- Regular workouts (3+ times per week) are associated with lower body fat percentages.
- Ensure sufficient hydration during workouts to improve performance and recovery.
- Customize fitness plans based on BMI and experience level for better outcomes.
""")