import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# 1. Start Logging
with mlflow.start_run():
    # --- Load Data ---
    df = pd.read_csv('data/churn.csv')
    
    # !!! FIX: Standardize column names !!!
    # The csv has 'gender', but our list uses 'Gender'. Let's rename it.
    df.rename(columns={'gender': 'Gender'}, inplace=True)
    
    # --- Preprocessing ---
    # Convert TotalCharges to number
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna() # Drop rows with missing values
    
    # Define Features (X) and Target (y)
    # We drop ID and the Target column
    # Note: 'Churn' is the target column in the specific CSV linked previously
    # If your CSV calls it 'Churn_Yes' or something else, change 'Churn' below
    X = df.drop(columns=['customerID', 'Churn']) 
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 2. Define "feature groups" for preprocessing
    # These names MUST match the columns in your CSV exactly
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Gender', 'Partner', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract', 
                            'PaperlessBilling', 'PaymentMethod']

    # 3. Create a Preprocessing Pipeline
    # Numeric cols -> Impute missing (avg) -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical cols -> Impute missing -> OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle them together
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Create the Final Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train (The pipeline handles the ugly matrix math for us!)
    print("Training Pipeline...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    accuracy = model_pipeline.score(X_test, y_test)
    print(f"Pipeline Accuracy: {accuracy:.4f}")

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save Model Locally (for the App) and to MLflow
    joblib.dump(model_pipeline, 'model.pkl')
    mlflow.sklearn.log_model(model_pipeline, "model")