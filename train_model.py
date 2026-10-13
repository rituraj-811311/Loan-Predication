import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = pd.read_csv("loan_data.csv")

print("First 5 rows:")
print(data.head())


# 2. EDA
print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())


# 3. Remove duplicates
data = data.drop_duplicates()


# 4. Feature Engineering
data["LoanIncomeRatio"] = data["LoanAmount"] / data["Income"]


# 5. Convert target into numerical form
data["LoanStatus"] = data["LoanStatus"].map({
    "Yes": 1,
    "No": 0
})


# 6. Input and output
X = data.drop("LoanStatus", axis=1)
y = data["LoanStatus"]


# 7. Feature types
numeric_features = [
    "Age",
    "Income",
    "CreditScore",
    "LoanAmount",
    "Experience",
    "LoanIncomeRatio"
]

categorical_features = [
    "Education",
    "Employment"
]


# 8. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)


# 9. Create ML pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])


# 10. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 11. Train model
print("\nTraining model...")
model.fit(X_train, y_train)


# 12. Prediction
y_pred = model.predict(X_test)


# 13. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 14. Save model
joblib.dump(model, "loan_model.pkl")

print("\nModel saved as loan_model.pkl")