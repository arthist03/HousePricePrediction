# Step 1: Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


# Step 2: Loading dataset
df = pd.read_csv('train.csv')

# Step 3: Handle Missing Data
features = ['OverallQual', 'GrLivArea',
            'GarageCars', 'TotalBsmtSF', 'SalePrice']
df = df[features]
df.dropna()


# Step 4: Split training and Testing data
x = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Step 5: Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Step 6: train Random Frorest Model for to train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Step 7: Evaluate Model Performace
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)


# Step 8: Save Trained Model
joblib.dump(model, 'house_price_model.pkl')
