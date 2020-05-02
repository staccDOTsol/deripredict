from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from time import sleep
import os
import pandas as pd
import deribit_wrapper
import requests
from flask import Flask
from flask import jsonify

from flask import flash, render_template, request, redirect

app = Flask(__name__)
app.secret_key = "coindexpoc_]K#)=fq;wAu-4zSu%xu}yer+/rw%(n"

import _thread

def flaskThread():
    app.run(debug=False, use_reloader=False,host='0.0.0.0', port=8089)

class DeriPredict( object ):
  def multivariate_data(self, dataset, target, start_index, end_index, history_size,
                        target_size, step, single_step=False):
    data = []
    labels = []

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

  def create_time_steps(self, length):
    return list(range(-length, 0))

  def show_plot(self, plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = self.create_time_steps(plot_data[0].shape[0])
    if delta:
      future = delta
    else:
      future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                 label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

  def plot_train_history(self, history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


  def run_live_predict(self, trained_model, step = 1, future_target = 0):
    hist_wrapper = deribit_wrapper.DeribitWrapper()
    BATCH_SIZE = 1
    df = hist_wrapper.fetch_chart_data(days_lookback=1)
    feature_cols = ["volume", "spread_roc", "spread"]   
    df["spread"] = df["high"] - df["low"]
    df["spread_roc"] = df["spread"].rolling(window=55).mean()
    df.dropna(inplace=True)
    print(df.iloc[-1])
    features = df[feature_cols].values 
    #print(features[-1])
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    dataset = (features - means) / stds
    past_history = 144
    start_idx = len(df) - (past_history+10)
    x_val_single, y_val_single = self.multivariate_data(dataset, dataset[:, 2],
                                                  start_idx, None, past_history,
                                                  future_target, step,
                                                  single_step=True)
    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    factor = 0
    for x, y in val_data_single.take(1):
      prediction = trained_model.predict(x)[0]
      denormed = (prediction * stds[2]) + means[2]
      target_denorm = (y_val_single[-1] * stds[2]) + means[2]
      last_spread = 0.0
      i = 1
      while last_spread == 0.0:
        last_spread = df["spread"].iloc[-i]
        i += 1
      factor = (last_spread - denormed) / last_spread
      if factor < 0.0:
        factor = factor * -1  
    return factor 


  def get_predictor(self, STEP = 1, future_target = 5):
    TRAIN_SPLIT = 6000
    BUFFER_SIZE = 455
    BATCH_SIZE = 255
    EVALUATION_INTERVAL = 20
    EPOCHS = 16
    tf.random.set_seed(13)

    feature_cols = ["volume", "spread_roc", "spread"]
    hist_wrapper = deribit_wrapper.DeribitWrapper()

    df = hist_wrapper.fetch_chart_data()
    df["spread"] = df["high"] - df["low"]
    df["spread_roc"] = df["spread"].rolling(window=55).mean()
    df.dropna(inplace=True)
    features = df[feature_cols].values
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    dataset = (features - means) / stds

    past_history = 144
    

    x_train_single, y_train_single = self.multivariate_data(dataset, dataset[:, 2], 0,
                                                      TRAIN_SPLIT, past_history,
                                                      future_target, STEP,
                                                      single_step=True)
    x_val_single, y_val_single = self.multivariate_data(dataset, dataset[:, 2],
                                                  TRAIN_SPLIT, None, past_history,
                                                  future_target, STEP,
                                                  single_step=True)

    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


    # setup model
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(21, return_sequences=True,
                                              input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.LSTM(13))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)
    '''
    for x, y in val_data_single.take(1):
      predicted = single_step_model.predict(x)[0]
      plot = self.show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                        predicted], 1,
                      'Single Step Prediction')
      plot.show()
    '''
    return single_step_model
  predictions = {}

  def prediction_service(self, predictors={}):
    if predictors == {}:
      predictors["1m"] = {
          "model" : self.get_predictor(1, 1),
          "step" : 1,
          "target_step": 1
      }
      predictors["5m"] = {
          "model" : self.get_predictor(1,5),
          "step" : 1,
          "target_step": 5
      } 
    for p in predictors:
      p_dict = predictors[p]
      self.predictions[p] = self.run_live_predict(p_dict["model"], p_dict["step"])
    time.sleep(60)
    self.prediction_service(predictors=predictors)
  def run(self):
    self.prediction_service()
  @app.route('/predictions', methods=['GET'])
  def set():
      print(dp.predictions)
      try:
          answer = {'1m':json.dumps(str(dp.predictions['1m'][-1])), '5m': json.dumps(str(dp.predictions['5m'][-1]))}
          
          return(json.dumps(answer))
      except Exception as e:
          print(e)
          return('500')
          
_thread.start_new_thread(flaskThread,())
dp = DeriPredict(  )
running = False
if __name__ == '__main__':
    while True:
      if not running:
        try:
            
            running = True
            dp.run()
            
        except( KeyboardInterrupt, SystemExit ):
            #print( "Cancelling open orders" )
            sys.exit()
        except Exception as e:
            print(e)
            running = False
      sleep(60)