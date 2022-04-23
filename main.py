# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

''' THIS CODE IS DIRECTLY FROM MY KAGGLE NOTEBOOK '''

import numpy as np # linear algebra
import matplotlib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

store_data = pd.read_csv("../input/retaildataset/Features data set.csv")
sales_data = pd.read_csv('../input/retaildataset/sales data-set.csv')

sales45_data = sales_data.iloc[421437:]
store45_data = store_data.iloc[8010:8143]

predictive_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']

X = store45_data[predictive_features]
y = sales45_data.Weekly_Sales

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

n = 55
retail_model = RandomForestRegressor(max_leaf_nodes=n, random_state=1)
retail_model.fit(train_X, train_y)

predicted_values = retail_model.predict(val_X)
val_mae = mean_absolute_error(predicted_values, val_y)

print("Validation MAE when using {0} max_leaf_nodes: {1}".format(n, val_mae))
matplotlib.pyplot.plot(train_X, train_y)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
