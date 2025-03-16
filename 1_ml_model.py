import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# sample data
data = pd.DataFrame({
    "YearsAtCompany": [1, 3, 5, 2, 4],
    "EmployeeSatisfaction": [0.01, 0.8, 0.5, 0.2, 0.9],
    "Position": ["Non-Manager", "Manager", "Non-Manager", "Manager", "Non-Manager"],
    "Salary": [4, 2, 3, 5, 1],
    "Target": [0, 1, 0, 1, 0]
})

# Encode categorical feature
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

# Features and Target
X = data.drop("Target", axis=1)
y = data["Target"]

# Train RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X,y)

# save model
joblib.dump(model, "rfmodel.pkl")
joblib.dump(le, "le.pkl")

print("Models are dumped successfully")