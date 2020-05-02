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
import ftx_funding_wrapper
import binance_funding_wrapper
import requests
from flask import Flask
from flask import jsonify
import sys
import linecache
from flask import flash, render_template, request, redirect

app = Flask(__name__)
app.secret_key = "coindexpoc_]K#)=fq;wAu-4zSu%xu}yer+/rw%(n"

import _thread
def PrintException():
  exc_type, exc_obj, tb = sys.exc_info()
  f = tb.tb_frame
  lineno = tb.tb_lineno
  filename = f.f_code.co_filename
  linecache.checkcache(filename)
  line = linecache.getline(filename, lineno, f.f_globals)
  string = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
  print(string) 
def flaskThread():
    app.run(debug=False, use_reloader=False,host='0.0.0.0', port=9292)

class DeriPredict( object ):
  rates = {}
  rates['binance'] = {}
  rates['ftx'] = {}
  binance = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex").json()  
  for rate in binance:
      rates['binance'][rate['symbol'].replace('USDT', '')] = float(rate['lastFundingRate']) * 3
  ftx = requests.get("https://ftx.com/api/funding_rates").json()['result']
  doneFtx = {}
  for rate in ftx:
      doneFtx[rate['future'].replace('-PERP', '')] = False
  coins = []
  for rate in ftx:
      if rate['future'].replace('-PERP', '') != 'BTC':
          if doneFtx[rate['future'].replace('-PERP', '')] == False:
              doneFtx[rate['future'].replace('-PERP', '')] = True
              if rate['future'].replace('-PERP', '') in rates['binance']:
                  if rate['future'].replace('-PERP', '') not in coins:
                    coins.append(rate['future'].replace('-PERP', ''))
              rates['ftx'][rate['future'].replace('-PERP', '')] = rate['rate'] * 24
  factors = {}
  def multivariate_data(self, dataset, target, start_index, end_index, history_size,
                        target_size, step, single_step=False):
    try:
      data = []
      labels = []

      start_index = start_index + history_size
      end_index  = start_index + len(dataset)
      #if end_index is None:
      #  end_index = len(dataset) - target_size
      end_index = len(dataset) - 1

      for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        try:
          data.append(dataset[indices])
        
          if single_step:
            labels.append(target[i+target_size])
          else:
            labels.append(target[i:i+target_size])
        except:
          return np.array(data), np.array(labels)
      return np.array(data), np.array(labels)
    except:
      PrintException()
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
    ftx_wrapper = ftx_funding_wrapper.DeribitWrapper()
    binance_wrapper = binance_funding_wrapper.DeribitWrapper()
    BATCH_SIZE = 2
    
    for coin in self.coins:
      df = ftx_wrapper.fetch_chart_data(days_lookback=1,instrument=coin)
      df2 = binance_wrapper.fetch_chart_data(days_lookback=1,instrument=coin)
      #print(df)
      #print(df2)

      for i in range(1, len(df)):
        df.loc[i, 'spread'] = df.loc[i-1, 'volume'] - df.loc[i, 'volume']
      for i in range(1, len(df2)):
        df2.loc[i, 'spread'] = df2.loc[i-1, 'volume'] - df2.loc[i, 'volume']
      feature_cols = ["volume", "spread_roc", "spread"]   
      df["spread_roc"] = df["spread"].rolling(window=55).mean()
      #df.dropna(inplace=True)
      #print(df.iloc[-1])
      features = df[feature_cols].values 
      df2["spread_roc"] = df2["spread"].rolling(window=55).mean()
      #df2.dropna(inplace=True)
      #print(df2.iloc[-1])
      features2 = df2[feature_cols].values 
      #print(features[-1])
      means = features.mean(axis=0)
      stds = features.std(axis=0)
      dataset = (features - means) / stds
      past_history = 144
      start_idx = 0#len(df) - (past_history+10)
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

      means2 = features2.mean(axis=0)
      stds2 = features2.std(axis=0)
      dataset2 = (features2 - means2) / stds2
      past_history2 = 144
      start_idx2 = 0#len(df2) - (past_history2+10)
      x_val_single2, y_val_single2 = self.multivariate_data(dataset2, dataset2[:, 2],
                                                    start_idx2, None, past_history2,
                                                    future_target, step,
                                                    single_step=True)

      val_data_single2 = tf.data.Dataset.from_tensor_slices((x_val_single2, y_val_single2))
      val_data_single2 = val_data_single.batch(BATCH_SIZE).repeat()
      factor2 = 0
      for x, y in val_data_single.take(1):
        prediction = trained_model.predict(x)[0]
        denormed = (prediction * stds2[2]) + means2[2]
        target_denorm = (y_val_single2[-1] * stds2[2]) + means2[2]
        last_spread = 0.0
        i = 1
        while last_spread == 0.0:
          last_spread = df2["spread"].iloc[-i]
          i += 1
        factor2 = (last_spread - denormed) / last_spread
        if factor2 < 0.0:
          factor2 = factor2 * -1  


      self.factors[coin] = {'ftx': factor, 'binance': factor2} 

     
  def get_predictor(self, STEP = 1, future_target = 2, coin='BTC'):
    try:
      TRAIN_SPLIT = 6000
      BUFFER_SIZE = 455
      BATCH_SIZE = 255
      EVALUATION_INTERVAL = 20
      EPOCHS = 1 #16
      tf.random.set_seed(13)

      feature_cols = ["volume", "spread_roc", "spread"]
      hist_wrapper = ftx_funding_wrapper.DeribitWrapper()
      
      single_step_models = {}
      df = hist_wrapper.fetch_chart_data(instrument=coin)
      
      for i in range(1, len(df)):
        df.loc[i, 'spread'] = df.loc[i-1, 'volume'] - df.loc[i, 'volume'] 
      df["spread_roc"] = df["spread"].rolling(window=55).mean()
      #df.dropna(inplace=True)
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
      done = False
      while not done:
        if len(x_train_single) > len(y_train_single):
          x_train_single = np.delete(x_train_single, [0])#new_a = np.delete(a, index)

        else:
          done = True
      done = False
      while not done:
        if len(x_val_single) > len(y_val_single):
          x_val_single = np.delete(x_val_single, [0])
        else:
          done = True
      print(len(x_train_single.shape))
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
                                                  validation_steps=10)
      '''
      for x, y in val_data_single.take(1):
        predicted = single_step_model.predict(x)[0]
        plot = self.show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          predicted], 1,
                        'Single Step Prediction')
        plot.show()
      '''
      return single_step_model
    except:
      PrintException()
  predictions = {}
  for coin in coins:
    predictions[coin] = {}
  def prediction_service(self, predictors={}):
    try:
      #print(predictors)
      if predictors == {}:
        #print(self.coins)
        for coin in self.coins:
          if coin not in predictors:
            predictors[coin] = {}

            predictors[coin]["1m"] = {
                "model" : self.get_predictor(1, 1, coin),
                "step" : 1,
                "target_step": 1
            }
            predictors[coin]["5m"] = {
                "model" : self.get_predictor(1,5, coin),
                "step" : 1,
                "target_step": 5
            } 
      for coin in predictors:
        for p in predictors[coin]:
          p_dict[coin] = predictors[coin][p]
          self.predictions[coin][p] = self.run_live_predict(p_dict[coin]["model"], p_dict[coin]["step"])
      
      self.prediction_service(predictors=predictors)
      time.sleep(5)# sleep(60)
    except:
      PrintException()
  def run(self):
    self.prediction_service()
  @app.route('/predictions', methods=['GET'])
  def set():
      print(dp.predictions)
      try:
          
          return(json.dumps(str(dp.predictions)))
      except Exception as e:
          PrintException()
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
      sleep(5)# sleep(60)