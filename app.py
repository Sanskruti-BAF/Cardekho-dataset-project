import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv(r'C:\Users\harsh\Downloads\WebApp-for-Prediction-of-Car-Price-CarDekho-Vehicle-Dataset-\cardata.csv')

# Basic preprocessing
df.drop_duplicates(inplace=True)
df['Selling_Price'] = df['Selling_Price'].fillna(df['Selling_Price'].mean())

# Drop the 'Car_Name' column
df = df.drop(['Car_Name'], axis=1)

# Define feature and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Correct Year feature
X['Year'] = 2020 - X['Year']

# Define the column transformer
num_features = ['Present_Price', 'Kms_Driven', 'Owner', 'Year']
cat_features = ['Fuel_Type', 'Seller_Type', 'Transmission']

column_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ],
    remainder='passthrough'
)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline (which includes the fitted transformer and the model)
with open('car_price_prediction_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pipeline (which includes the fitted transformer and the model)
pipeline = pickle.load(open('car_price_prediction_pipeline.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        
        Fuel_Type = request.form['Fuel_Type_Petrol']
        Seller_Type = request.form['Seller_Type_Individual']
        Transmission = request.form['Transmission_Mannual']
        
        input_data = pd.DataFrame([[Present_Price, Kms_Driven, Owner, Year, Fuel_Type, Seller_Type, Transmission]],
                                  columns=['Present_Price', 'Kms_Driven', 'Owner', 'Year', 'Fuel_Type', 'Seller_Type', 'Transmission'])
        
        # Correct Year feature
        input_data['Year'] = 2020 - input_data['Year']
        
        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)
        output = round(prediction[0], 2)
        
        if output < 0:
            return render_template('index.html', prediction_text="Sorry, you are not eligible to sell this car.")
        else:
            return render_template('index.html', prediction_text="You can sell the car at ₹{}".format(output))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
