from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from numpy.random import normal
import cPickle
from datetime import datetime


#My fnctions
import data_prep as dp
import run_random_forest as rrf
import project_specific_functions as psf

file_path = '~/Desktop/Kaggle/Liberty/'
model_name = 'rf_model_{}'.format(datetime.strftime(datetime.now(), '%Y_%m_%d_%H:%M:%S'))
print(model_name)

# Read Data
data = dp.read_data('{}data/train.csv'.format(file_path))

data_full = data.copy()
y_data_full, X_data_full = dp.get_target_variable(data_full, 'Hazard')

train, test = dp.split_train_test(data)
y_train, X_train = dp.get_target_variable(train, 'Hazard')
y_test, X_test = dp.get_target_variable(test, 'Hazard')

y_train = pd.Series(psf.adjust_y(y_train))
y_test= pd.Series(psf.adjust_y(y_test))

X_ID = train.pop('Id')
Y_ID = test.pop('Id')

# are there any missing values?
dp.print_columns_with_missing(X_train)
dp.print_columns_with_missing(X_test)

# Run the model
rf_model = rrf.run_random_forest(X_train, y_train, X_test, y_test)
print(rf_model.get_params())
rf_final_model = rf_model.fit(X_data_full, y_data_full.values)


# Score the actual test set
test = dp.read_data('{}data/test.csv'.format(file_path))
test = test.set_index('Id')
test = dp.factorize_variables(test)
test_predictions = rf_final_model.predict(test)
test = pd.DataFrame(np.transpose([test.index, test_predictions]))
test.columns = ["Id", "Hazard"]



# Store results and pickle model
test.to_csv('{}Output/{}.csv'.format(file_path, model_name), drop = True)
#with open('/Users/Adrianna/Desktop/Kaggle/Liberty/Output/rf.pkl', 'wb') as f:
#	cPickle.dump(rf_model, f)
