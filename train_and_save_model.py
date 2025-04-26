import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv("fatty_liver.csv")  # Make sure this file is in the same folder

# Define X and y
X = data.drop(["id", "status"], axis=1)
y = data["status"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model to model.pkl
joblib.dump(rf_model, "model.pkl")

print("✅ Model trained and saved as 'model.pkl'")


