import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import pickle

#Import data
cpu_train_a= 'https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv'

cpu_train_b= 'https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv'

cpu_test_a='https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv'

cpu_test_b='https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv'

#Read with pandas

train_a= pd.read_csv(cpu_train_a)
test_a = pd.read_csv(cpu_test_a)

train_b = pd.read_csv(cpu_train_b)
test_b = pd.read_csv(cpu_test_b)


#Preprocess

train_a['datetime'] = pd.to_datetime(train_a['datetime'])
train_a = train_a.set_index('datetime')

test_a['datetime'] = pd.to_datetime(test_a['datetime'])
test_a = test_a.set_index('datetime')

train_b['datetime'] = pd.to_datetime(train_b['datetime'])
train_b = train_b.set_index('datetime')

test_b['datetime'] = pd.to_datetime(test_b['datetime'])
test_b = test_b.set_index('datetime')

#Fit model a
model = ARIMA(train_a, order=(6,2,1))
model_fit = model.fit()

#Save

filename = 'models/model_a.sav'

pickle.dump(model, open(filename,'wb'))

#Fit model b
model_b = ARIMA(train_b, order=(6,2,1))

modelb_fit = model_b.fit()

#Save 
filename = 'models/model_b.sav'

pickle.dump(model_b, open(filename,'wb'))

