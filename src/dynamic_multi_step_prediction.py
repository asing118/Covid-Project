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


df = pd.read_csv('../data/manual/processed_dynamic_data_US.csv', header=0, parse_dates=[0])

print(df.head())


print(df.isnull().sum())

TRAIN_SPLIT = 128
tf.random.set_seed(92)

def create_time_steps(length):
  return list(range(-length, 0))
  
  

features = df.loc[:, df.columns != 'date']
features.index = df['date']
print(features.head)


features.plot(subplots=True)



dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)



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

EVALUATION_INTERVAL = 38
EPOCHS = 50
BATCH_SIZE = 4
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

multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          dropout = 0.2,
                                          input_shape=x_train_multi.shape[-2:]))                                      
multi_step_model.add(tf.keras.layers.LSTM(32, activation='relu',dropout=0.2))
multi_step_model.add((tf.keras.layers.Dense(14)))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mse')


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
  #multi_step_plot(((x[0]*data_std[18])+data_mean[18]), ((y[0]*data_std[18])+data_mean[18]), ((multi_step_model.predict(x)[0] * data_std[18]) + data_mean[18]))  
  #plt.show()  

print("average rmse error", np.mean(rmse_error_list)) 
  
  #multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])    


'''

save model

'''

multi_step_model.save('../checkpoint/model_US_dynamic_data.h5') 
  

