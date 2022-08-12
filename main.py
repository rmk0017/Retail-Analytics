
''' THIS CODE IS FROM MY KAGGLE NOTEBOOK 
I decided to use a random forest because it averages the predictions of each component decision trees, leading to more accurate predictions.
'''

#Importing required libraries and tools
import numpy as np 
import matplotlib
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Extracting data from CSVs and accessing relevant data
store_data = pd.read_csv("../input/retaildataset/Features data set.csv")
sales_data = pd.read_csv('../input/retaildataset/sales data-set.csv')
sales45_data = sales_data.iloc[421437:]
store45_data = store_data.iloc[8010:8143]

#predictive_features is a list of all factors that influence retail sales
predictive_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']
X = store45_data[predictive_features]
y = sales45_data.Weekly_Sales

#Splitting the data we have acquired into training data and validation/testing data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

n = 55
retail_model = RandomForestRegressor(max_leaf_nodes=n, random_state=1)
retail_model.fit(train_X, train_y)

predicted_values = retail_model.predict(val_X)
val_mae = mean_absolute_error(predicted_values, val_y)

print("Validation MAE when using {0} max_leaf_nodes: {1}".format(n, val_mae))
matplotlib.pyplot.plot(train_X, train_y)
