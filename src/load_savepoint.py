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

start_date = '2020-05-13    '  #  YYYY-MM-DD
input_list = [0,1,0,1,0,1,0,1]
state_name = 'US'


def load_savepoint(start_date, state_name, input_list,param):
    
    df = pd.read_csv('./data/manual/processed_hybrid_data_%s.csv'%(state_name), header=0, parse_dates=[0])

    date_index = (df[df['date']==start_date].index.values)
    

    TRAIN_SPLIT = date_index[0]
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
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 18], TRAIN_SPLIT-past_history,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 18],
                                                 TRAIN_SPLIT-past_history, TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
                                                 
         
    new_model = tf.keras.models.load_model('./checkpoint/model_%s_hybrid_data.h5'%(state_name))
    
    true_val = y_val_multi
    pred_val = new_model.predict(x_val_multi)
    m = tf.keras.metrics.RootMeanSquaredError()
    _ = m.update_state(true_val,pred_val)
    #multi_step_plot(x_val_multi, y_val_multi, pred_val) 
    
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
    _x_train_multi, _y_train_multi = multivariate_data(dataset, dataset[:, 18], TRAIN_SPLIT-past_history,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    _x_val_multi, _y_val_multi = multivariate_data(dataset, dataset[:, 18],
                                                 TRAIN_SPLIT-past_history, TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
                                                 
         
    _new_model = tf.keras.models.load_model('./checkpoint/model_%s_hybrid_data.h5'%(state_name))
    
    true_val_updated = _y_val_multi
    _pred_val = _new_model.predict(_x_val_multi)
    m = tf.keras.metrics.RootMeanSquaredError()
    _ = m.update_state(true_val_updated,_pred_val) 
    multi_step_plot_new_pred(_x_val_multi, _y_val_multi, pred_val,_pred_val)     
    
    
    
    
    
    
STEP = 1    
    
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  indices = range(start_index, end_index, step)
  data.append(dataset[indices])


  labels.append(target[start_index:start_index+target_size])

  return np.array(data), np.array(labels)    
    
    
def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  history = np.squeeze(history, axis=0)
  true_future = np.squeeze(true_future, axis=0)
  prediction = np.squeeze(prediction, axis=0)
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
    
 
#load_savepoint(start_date, state_name, input_list,'retail_and_recreation_percent_change_from_baseline')