# Import required libraries
# streamlit -> used to create the web interface
# pandas -> used for dataset handling
# matplotlib -> used for visualization
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Tools for splitting dataset and evaluating models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------
# Load Dataset
# -------------------------

# Load dataset from CSV file
dataset = pd.read_csv("dataset.csv")

# Separate features (inputs) and target (output label)
features = dataset.drop(columns=["anxiety"])   # input columns
target = dataset["anxiety"]                    # output column

# Split dataset into training and testing data
# 75% for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=42
)

# -------------------------
# Train Models
# -------------------------

# Initialize three machine learning models
tree_model = DecisionTreeClassifier()      # Decision Tree algorithm
forest_model = RandomForestClassifier()    # Random Forest algorithm
log_model = LogisticRegression(max_iter=1000)  # Logistic Regression algorithm

# Train the models using training data
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

# Calculate prediction accuracy for each model
tree_score = accuracy_score(y_test, tree_model.predict(X_test))
forest_score = accuracy_score(y_test, forest_model.predict(X_test))
log_score = accuracy_score(y_test, log_model.predict(X_test))


# -------------------------
# Streamlit Interface
# -------------------------

# Title displayed on the web application
st.title("Student Exam Stress Predictor")

# Short description for users
st.write("Provide the details below to estimate exam anxiety level.")

# Slider input for sleep hours
sleep_hours = st.slider("Sleep Duration (hours)", 0, 10, 6)

# Slider input for study hours
study_hours = st.slider("Daily Study Time", 0, 10, 4)

# Dropdown inputs for symptoms related to exam anxiety
nervous_input = st.selectbox("Do you feel nervous before exam?", ["No", "Yes"])
heart_input = st.selectbox("Heart rate increases before exam?", ["No", "Yes"])
focus_input = st.selectbox("Difficulty concentrating?", ["No", "Yes"])

# Convert user text inputs into numeric values (required for ML models)
nervous_val = 1 if nervous_input == "Yes" else 0
heart_val = 1 if heart_input == "Yes" else 0
focus_val = 1 if focus_input == "Yes" else 0

# Create a dataframe from user input for prediction
user_data = pd.DataFrame(
    [[sleep_hours, study_hours, nervous_val, heart_val, focus_val]],
    columns=["sleep", "study", "nervous", "heart", "focus"]
)


# -------------------------
# Prediction Button
# -------------------------

# When user clicks the button, model will predict anxiety level
if st.button("Check Anxiety Level"):

    # Use Random Forest model for prediction
    result = forest_model.predict(user_data)[0]

    # Display predicted anxiety level
    st.subheader("Predicted Anxiety Level")
    st.success(result)

    # Show suggestions based on prediction
    if result == "High":
        st.warning("Advice: Take small breaks and practice relaxation.")
    elif result == "Medium":
        st.info("Advice: Stay calm and maintain good sleep.")
    else:
        st.success("Advice: You appear relaxed. Continue preparation.")


# -------------------------
# Accuracy Comparison
# -------------------------

# Section title for model accuracy comparison
st.subheader("Algorithm Accuracy")

# Create a dataframe containing accuracy of all models
comparison = pd.DataFrame({
    "Algorithm": ["Decision Tree", "Random Forest", "Logistic Regression"],
    "Accuracy": [tree_score, forest_score, log_score]
})

# Display the comparison table
st.write(comparison)

# Create a bar chart to visualize model performance
fig, ax = plt.subplots()
ax.bar(comparison["Algorithm"], comparison["Accuracy"])
ax.set_ylabel("Accuracy Score")
ax.set_title("Model Comparison")

# Display chart in Streamlit
st.pyplot(fig)