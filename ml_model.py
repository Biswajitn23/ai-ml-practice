import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/students.csv")

# Handle missing values (future-safe)
for col in ["Maths", "Science", "English"]:
    df[col] = df[col].fillna(df[col].mean())

# Feature engineering
df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

# Features and target
X = df[["Maths", "Science", "English"]]
y = df["Average"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
