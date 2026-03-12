import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------
# Load Dataset
# -------------------------

dataset = pd.read_csv("dataset.csv")

features = dataset.drop(columns=["anxiety"])
target = dataset["anxiety"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=42
)

# -------------------------
# Train Models
# -------------------------

tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()
log_model = LogisticRegression(max_iter=1000)

tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

tree_score = accuracy_score(y_test, tree_model.predict(X_test))
forest_score = accuracy_score(y_test, forest_model.predict(X_test))
log_score = accuracy_score(y_test, log_model.predict(X_test))


# -------------------------
# Streamlit Interface
# -------------------------

st.title("Student Exam Stress Predictor")

st.write("Provide the details below to estimate exam anxiety level.")

sleep_hours = st.slider("Sleep Duration (hours)", 0, 10, 6)
study_hours = st.slider("Daily Study Time", 0, 10, 4)

nervous_input = st.selectbox("Do you feel nervous before exam?", ["No", "Yes"])
heart_input = st.selectbox("Heart rate increases before exam?", ["No", "Yes"])
focus_input = st.selectbox("Difficulty concentrating?", ["No", "Yes"])

# convert text to numeric
nervous_val = 1 if nervous_input == "Yes" else 0
heart_val = 1 if heart_input == "Yes" else 0
focus_val = 1 if focus_input == "Yes" else 0

# create dataframe for prediction
user_data = pd.DataFrame(
    [[sleep_hours, study_hours, nervous_val, heart_val, focus_val]],
    columns=["sleep", "study", "nervous", "heart", "focus"]
)


# -------------------------
# Prediction Button
# -------------------------

if st.button("Check Anxiety Level"):

    result = forest_model.predict(user_data)[0]

    st.subheader("Predicted Anxiety Level")
    st.success(result)

    if result == "High":
        st.warning("Advice: Take small breaks and practice relaxation.")
    elif result == "Medium":
        st.info("Advice: Stay calm and maintain good sleep.")
    else:
        st.success("Advice: You appear relaxed. Continue preparation.")


# -------------------------
# Accuracy Comparison
# -------------------------

st.subheader("Algorithm Accuracy")

comparison = pd.DataFrame({
    "Algorithm": ["Decision Tree", "Random Forest", "Logistic Regression"],
    "Accuracy": [tree_score, forest_score, log_score]
})

st.write(comparison)

fig, ax = plt.subplots()
ax.bar(comparison["Algorithm"], comparison["Accuracy"])
ax.set_ylabel("Accuracy Score")
ax.set_title("Model Comparison")

st.pyplot(fig)