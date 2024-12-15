import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'C:\Users\harsh\Downloads\WebApp-for-Prediction-of-Car-Price-CarDekho-Vehicle-Dataset-\cardata.csv')

# Display the first few rows of the dataset
print(df.head())

# Understand the target variable
print(df['Selling_Price'].describe())

# Check for duplicates
print(f'Duplicate rows: {df.duplicated().sum()}')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check for data inconsistencies
print(df.info())
print(df.describe(include='all'))

# Check for missing values
print(df.isnull().sum())

# Treat missing values
imputer = SimpleImputer(strategy='mean')
df['Selling_Price'] = imputer.fit_transform(df[['Selling_Price']])

# Exploratory Data Analysis (EDA)
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

# Scaling (Standardization/Normalization)
scaler = StandardScaler()
df[['Present_Price', 'Kms_Driven']] = scaler.fit_transform(df[['Present_Price', 'Kms_Driven']])

# Encoding (Converting categorical variables)
df = pd.get_dummies(df, drop_first=True)

# Split the data into training and testing sets
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply multiple machine learning models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'R2 Score': r2}

# Compare the accuracy of each model
results_df = pd.DataFrame(results).T
print(results_df)

# Target accuracy greater than 80%
target_accuracy_models = results_df[results_df['R2 Score'] > 0.8]
print('Models with R2 Score greater than 0.8:')
print(target_accuracy_models)

# Predicting car prices using the best model (e.g., Random Forest)
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

# Print sample predictions
print("Sample Predictions:")
print(predictions[:10])
