import tensorflow as tf

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


df = pd.read_csv('../data/manual/processed_hybrid_data_US.csv', header=0, parse_dates=[0])
#df_static = 
print(df.head())


print(df.isnull().sum())

TRAIN_SPLIT = 128
tf.random.set_seed(92)

def create_time_steps(length):
  return list(range(-length, 0))
  
  
#features_considered = ['cases', 'deaths']
#features = df[features_considered]
features = df.loc[:, df.columns != 'date']
features.index = df['date']
print(features.head)


features.plot(subplots=True)
#plt.show()


dataset = features.values
data_mean = dataset[:TRAIN_SPLIT,:19].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT,:19].std(axis=0)

std_append = np.repeat(1.0,36)
mean_append = np.repeat(0.0,36)

data_mean = np.concatenate([data_mean,mean_append])
data_std = np.concatenate([data_std,std_append])

print(data_std)


dataset = (dataset-data_mean)/data_std



def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []
  print("length of dataset", len(dataset))

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)
  
  
past_history = 7
STEP = 1

EVALUATION_INTERVAL = 64
EPOCHS = 100
BATCH_SIZE = 2
BUFFER_SIZE = 8

future_target = 14
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 18], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 18],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))


train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), 'b', label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b', linestyle=':',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r', linestyle=':',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()
  
  
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()  
  
def last_time_step_mse(Y_true, Y_pred):
    return tf.keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])  

multi_step_model = tf.keras.models.Sequential()
#tf.keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
#                        input_shape=[None, 1]),
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          dropout = 0.2,
                                          input_shape=x_train_multi.shape[-2:]))
#multi_step_model.add(Dropout(0.2))                                          
multi_step_model.add(tf.keras.layers.LSTM(32, activation='relu',dropout=0.2))
#multi_step_model.add(tf.keras.layers.LSTM(32, activation='relu',dropout=0.2))
#multi_step_model.add(Dropout(0.2))  
multi_step_model.add((tf.keras.layers.Dense(14)))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mse', learning_rate=0.0001)
#clipvalue=1.0

for x, y in val_data_multi.take(1):
  print (multi_step_model.predict(x).shape)
  
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=5)
                                          
                                          
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss') 


rmse_error_list = []
for x, y in val_data_multi.take(10):
  print(y.shape)
  true = (y[0])
  pred = (multi_step_model.predict(x)[0])
  m = tf.keras.metrics.RootMeanSquaredError()
  _ = m.update_state(true,pred)
  print("RMSE", m.result().numpy())
  rmse_error_list.append(m.result().numpy())
  multi_step_plot(((x[0]*data_std[18])+data_mean[18]), ((y[0]*data_std[18])+data_mean[18]), ((multi_step_model.predict(x)[0] * data_std[18]) + data_mean[18]))  
  plt.show()  

print("average rmse error", np.mean(rmse_error_list)) 
  
  #multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])    


'''

save model

'''

multi_step_model.save('../checkpoint/model_US_hybrid_data.h5')   

#######################################################################
######################################################################

###################################

start_date = '2020-05-13    '  #  YYYY-MM-DD
input_list = [0,1,0,1,0,1,0,1]
state_name = 'AZ'
param = 'retail_and_recreation_percent_change_from_baseline'
STEP = 1    
    
def multivariate_data_updated(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  indices = range(start_index, end_index, step)
  data.append(dataset[indices])


  labels.append(target[start_index:start_index+target_size])

  return np.array(data), np.array(labels)    
    

  
  
def multi_step_plot_new_pred(history, true_future, prediction, new_pred):
  plt.figure(figsize=(12, 6))
  history = np.squeeze(history, axis=0)
  true_future = np.squeeze(true_future, axis=0)
  prediction = np.squeeze(prediction, axis=0)
  new_pred = np.squeeze(new_pred, axis=0)
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), 'b', label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b', linestyle=':',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r', linestyle=':',
             label='Predicted Future')
  plt.plot(np.arange(num_out)/STEP, np.array(new_pred), 'g', linestyle=':',
             label='Updated Prediction')
  plt.legend(loc='upper left')
  plt.show()
  
def create_time_steps(length):
  return list(range(-length, 0))


start_date = '2020-06-10    '  #  YYYY-MM-DD
input_list = [0,1,0,1,0,1,0,1]
state_name = 'AZ'

past_history = 7
STEP = 1

future_target = 14
x_train_multi, y_train_multi = multivariate_data_updated(dataset, dataset[:, 18],                                                 TRAIN_SPLIT-past_history,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data_updated(dataset, dataset[:, 18],
                                             TRAIN_SPLIT-past_history, TRAIN_SPLIT, past_history,
                                             future_target, STEP)
                                             
     
new_model = tf.keras.models.load_model('../checkpoint/model_US_hybrid_data.h5')

true_val = y_val_multi
pred_val = new_model.predict(x_val_multi)
m = tf.keras.metrics.RootMeanSquaredError()
_ = m.update_state(true_val,pred_val)


indices = np.linspace((TRAIN_SPLIT - past_history), TRAIN_SPLIT-1, 7)

for i in range(len(input_list)):
    i = int(i)
    df.loc[indices[i:i+1],param] = input_list[i]
    
     
features = df.loc[:, df.columns != 'date']
features.index = df['date']

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT,:19].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT,:19].std(axis=0)

std_append = np.repeat(1.0,36)
mean_append = np.repeat(0.0,36)

data_mean = np.concatenate([data_mean,mean_append])
data_std = np.concatenate([data_std,std_append])    
dataset = (dataset-data_mean)/data_std





past_history = 7
STEP = 1

future_target = 14
_x_train_multi, _y_train_multi = multivariate_data_updated(dataset, dataset[:, 18], TRAIN_SPLIT-past_history,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
_x_val_multi, _y_val_multi = multivariate_data_updated(dataset, dataset[:, 18],
                                             TRAIN_SPLIT-past_history, TRAIN_SPLIT, past_history,
                                             future_target, STEP)
                                             
     
_new_model = tf.keras.models.load_model('../checkpoint/model_US_hybrid_data.h5')

true_val_updated = _y_val_multi
_pred_val = _new_model.predict(_x_val_multi)
m = tf.keras.metrics.RootMeanSquaredError()
_ = m.update_state(true_val_updated,_pred_val) 
multi_step_plot_new_pred(_x_val_multi, _y_val_multi, pred_val,_pred_val)     



#
#
#new_model = tf.keras.models.load_model('my_model.h5')
#new_model.summary()
#
#_rmse_error_list = []
#for x, y in val_data_multi.take(10):
#  print(y.shape)
#  true = (y[0])
#  pred = (new_model.predict(x)[0])
#  m = tf.keras.metrics.RootMeanSquaredError()
#  _ = m.update_state(true,pred)
#  print("RMSE", m.result().numpy())
#  _rmse_error_list.append(m.result().numpy())
#  #multi_step_plot(((x[0]*data_std[18])+data_mean[18]), ((y[0]*data_std[18])+data_mean[18]), ((multi_step_model.predict(x)[0] * data_std[18]) + data_mean[18]))  
#  #plt.show()  
#
#print("average rmse error", np.mean(_rmse_error_list)) 

