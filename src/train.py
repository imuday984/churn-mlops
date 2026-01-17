import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
# We refer to the data folder relative to the root
df = pd.read_csv('data/churn.csv')

# 2. Preprocessing (Simple for Phase 1)
# Convert 'TotalCharges' to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Drop customerID as it's not a feature
df = df.drop(columns=['customerID'])

# Convert categorical variables into numbers (Yes/No -> 1/0)
# This is a quick way to handle all string columns
df = pd.get_dummies(df, drop_first=True)

# 3. Define X and y
# Assume 'Churn_Yes' is the target after get_dummies
X = df.drop(columns=['Churn_Yes']) 
y = df['Churn_Yes']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Training Complete.")
print(f"Accuracy: {accuracy:.4f}")

# 7. Save Model (Legacy method - we will change this later)
joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")