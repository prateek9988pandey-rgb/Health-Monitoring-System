import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# --------------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------------

file_path = "/mnt/data/dataset_extracted/Health Monitor Dataset.xlsx"

df = pd.read_excel(file_path)

print("Dataset Loaded Successfully!")
print(df.head())

# --------------------------------------------------------
# 2. Basic Cleaning
# --------------------------------------------------------

# Drop rows with missing target
TARGET_COLUMN = "health_status"   # <- change to correct column name

df = df.dropna(subset=[TARGET_COLUMN])

# Fill missing numerical values
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Encode categoricals
df = pd.get_dummies(df, drop_first=True)

# --------------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------------

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------
# 4. Feature Scaling
# --------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------
# 5. Model Training (Random Forest)
# --------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# --------------------------------------------------------
# 6. Evaluation
# --------------------------------------------------------

y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------------
# 7. Save Model + Scaler
# --------------------------------------------------------

joblib.dump(model, "/mnt/data/health_monitor_model.pkl")
joblib.dump(scaler, "/mnt/data/health_scaler.pkl")

print("\nModel Saved Successfully!")
