<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load the dataset
df = pd.read_csv("student-scores.csv")

# Create total_score and average_score as done in the notebook
score_columns = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
df["total_score"] = df[score_columns].sum(axis=1)
df["average_score"] = df["total_score"] / 7

# Select features and target
features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities',
            'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
            'chemistry_score', 'biology_score', 'english_score', 'geography_score',
            'total_score', 'average_score']
target = 'career_aspiration'

X = df[features].copy()
y = df[target].copy()

# Encode categorical features
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender']) # male: 1, female: 0 (or vice versa, let's check)
# In untitled37.py: gender_encoded = 1 if gender.lower() == 'female' else 0
# So let's manually encode to match that:
X['gender'] = X['gender'].apply(lambda x: 1 if str(x).lower() == 'female' else 0)

X['part_time_job'] = X['part_time_job'].astype(int)
X['extracurricular_activities'] = X['extracurricular_activities'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Ensure the Models directory exists
if not os.path.exists('Models'):
    os.makedirs('Models')

# Save the model and scaler
with open("Models/model.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("Models/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully in Models/ directory.")
print("Class names in model:", model.classes_)
=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load the dataset
df = pd.read_csv("student-scores.csv")

# Create total_score and average_score as done in the notebook
score_columns = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
df["total_score"] = df[score_columns].sum(axis=1)
df["average_score"] = df["total_score"] / 7

# Select features and target
features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities',
            'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
            'chemistry_score', 'biology_score', 'english_score', 'geography_score',
            'total_score', 'average_score']
target = 'career_aspiration'

X = df[features].copy()
y = df[target].copy()

# Encode categorical features
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender']) # male: 1, female: 0 (or vice versa, let's check)
# In untitled37.py: gender_encoded = 1 if gender.lower() == 'female' else 0
# So let's manually encode to match that:
X['gender'] = X['gender'].apply(lambda x: 1 if str(x).lower() == 'female' else 0)

X['part_time_job'] = X['part_time_job'].astype(int)
X['extracurricular_activities'] = X['extracurricular_activities'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Ensure the Models directory exists
if not os.path.exists('Models'):
    os.makedirs('Models')

# Save the model and scaler
with open("Models/model.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("Models/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully in Models/ directory.")
print("Class names in model:", model.classes_)
>>>>>>> 60c17094c30019bfa7614b3031154f5fe540be04
