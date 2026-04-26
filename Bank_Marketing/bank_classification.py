import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATA 
# Using a specific separator if needed; usually, bank.csv is comma-separated
df = pd.read_csv("bank.csv", sep=",")

# 2. PREPROCESSING
# Convert target variable to binary (1 for 'yes', 0 for 'no')
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

# Separate features by type to handle them correctly
# We use 'include=object' to find text columns for One-Hot Encoding
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols_names = df.select_dtypes(include=["object"]).columns

# One-Hot Encoding: Converts text categories into binary columns (0 or 1)
# drop_first=True prevents the 'dummy variable trap' (multi-collinearity)
categorical_encoded = pd.get_dummies(df[categorical_cols_names], drop_first=True)

# Combine numerical features and newly encoded categorical features
df_final = pd.concat([df[numerical_cols], categorical_encoded], axis=1)

# 3. DATA SPLITTING
# X = Features, y = Target
X = df_final.drop("deposit", axis=1)
y = df_final["deposit"]

# Using a fixed random_state ensures results are reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2014
)

# 4. MODEL TRAINING
# n_estimators: Number of trees in the forest
# learning_rate: Controls how much we adjust weights per tree (shinkage)
model = XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.1, 
    n_jobs=-1, # -1 uses all available CPU cores for faster training
    random_state=2014
)
model.fit(X_train, y_train)

# 5. EVALUATION
pred = model.predict(X_test)

# Print metrics to understand Precision (false alarms) and Recall (missed deposits)
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
