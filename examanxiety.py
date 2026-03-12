
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# -------------------------
# Load CSV Dataset
# -------------------------

dataset = pd.read_csv("dataset.csv")

X = dataset[["sleep", "study", "nervous", "heart", "focus"]]
y = dataset["anxiety"]

classifier = DecisionTreeClassifier()
classifier.fit(X, y)


print("===== Exam Stress Predictor =====")

# -------------------------
# User Inputs
# -------------------------

sleep_hours = int(input("Enter sleep hours (0-10): "))
study_hours = int(input("Enter study hours (0-10): "))

nervous = input("Feeling nervous before exam? (yes/no): ")
heart = input("Heart beating fast? (yes/no): ")
focus = input("Trouble concentrating? (yes/no): ")

nervous = 1 if nervous.lower() == "yes" else 0
heart = 1 if heart.lower() == "yes" else 0
focus = 1 if focus.lower() == "yes" else 0


# -------------------------
# Prediction
# -------------------------

input_values = [[sleep_hours, study_hours, nervous, heart, focus]]

prediction = classifier.predict(input_values)[0]

print("\nEstimated Anxiety Level:", prediction)

if prediction == "High":
    print("Tip: Relaxation exercises and short breaks can help.")
elif prediction == "Medium":
    print("Tip: Stay consistent with study and sleep routine.")
else:
    print("Tip: You seem calm. Keep up the good preparation.")