import sys
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import pickle

train_datafile, test_datafile = sys.argv[1],sys.argv[2] if len(sys.argv) > 2 else print(" No input for train and test files")

train_df = pd.read_csv(train_datafile,sep=",")
test_df = pd.read_csv(test_datafile)

X_train = train_df.iloc[:,0:-1].values
y_train= np.array(train_df.iloc[:,-1])

X_test = test_df.iloc[:,0:-1].values
y_test = np.array(test_df.iloc[:,-1])

# Parameters
n_estimators = 5
random_state = 100
min_samples_leaf = 5
regressor = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state, min_samples_leaf=min_samples_leaf)

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

# Model evaluation
mae =  metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# Save model
project_path = os.getcwd()
filename = project_path+"/model/housing_model.sav"

with open(filename,'wb') as model_file:
    pickle.dump(regressor, model_file)



