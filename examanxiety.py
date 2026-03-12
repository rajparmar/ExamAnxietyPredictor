# Import pandas library to handle CSV data
import pandas as pd

# Import Decision Tree algorithm from sklearn
from sklearn.tree import DecisionTreeClassifier


# -------------------------
# Load CSV Dataset
# -------------------------

# Read the dataset from the CSV file
dataset = pd.read_csv("dataset.csv")

# X contains input features used for prediction
# These represent student behavior and symptoms
X = dataset[["sleep", "study", "nervous", "heart", "focus"]]

# y contains the target variable (anxiety level)
y = dataset["anxiety"]

# Create a Decision Tree model
classifier = DecisionTreeClassifier()

# Train the model using dataset
classifier.fit(X, y)


# Display program title
print("===== Exam Stress Predictor =====")

# -------------------------
# User Inputs
# -------------------------

# Ask the user to enter sleep hours
sleep_hours = int(input("Enter sleep hours (0-10): "))

# Ask the user to enter daily study hours
study_hours = int(input("Enter study hours (0-10): "))

# Ask if the student feels nervous before exams
nervous = input("Feeling nervous before exam? (yes/no): ")

# Ask if heart rate increases before exams
heart = input("Heart beating fast? (yes/no): ")

# Ask if the student has difficulty concentrating
focus = input("Trouble concentrating? (yes/no): ")

# Convert text inputs into numeric values
# yes = 1, no = 0 (required for ML model)
nervous = 1 if nervous.lower() == "yes" else 0
heart = 1 if heart.lower() == "yes" else 0
focus = 1 if focus.lower() == "yes" else 0


# -------------------------
# Prediction
# -------------------------

# Create input data in the same format as the training dataset
input_values = [[sleep_hours, study_hours, nervous, heart, focus]]

# Predict anxiety level using trained Decision Tree model
prediction = classifier.predict(input_values)[0]

# Display predicted anxiety level
print("\nEstimated Anxiety Level:", prediction)

# Provide suggestions based on predicted anxiety level
if prediction == "High":
    print("Tip: Relaxation exercises and short breaks can help.")
elif prediction == "Medium":
    print("Tip: Stay consistent with study and sleep routine.")
else:
    print("Tip: You seem calm. Keep up the good preparation.")