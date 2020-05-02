import requests
import json
import pandas as pd
import os
import numpy as np
import time
import datetime as dt
from statistics import mode

class DeribitWrapper(object):
    rates = {}
    rates['binance'] = {}
    rates['ftx'] = {}
    candlestick_endpoint = "/get_tradingview_chart_data?end_timestamp={}&instrument_name={}&resolution={}&start_timestamp={}"
    base_endpoint = "https://test.deribit.com/api/v2/public"
    info_endpoint = "/api/v3/exchangeInfo"
    volatility_endpoint = "/get_historical_volatility?currency={}"
    syms_endpoint = "/get_instruments"
    live_dir = "./live_data/deribit/"
    train_dir = "./hist_data/deribit/"
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
    


    
    def __init__(self):
        return

    def fetch_chart_data(self, instrument="BTC-PERPETUAL", interval=1, days_lookback=60):
        try:
            print(instrument)
            start, end = self.get_start_end(days_lookback)
            print(start, end)
            print("https://fapi.binance.com/fapi/v1/fundingRate?limit=1000&symbol=" + instrument + 'USDT')
            rates = requests.get("https://fapi.binance.com/fapi/v1/fundingRate?limit=1000&symbol=" + instrument + 'USDT').json()
            rates = list(reversed(rates))
            result = {}
            for rate in rates:
                #print(rate['symbol'])
                coin = rate['symbol'].replace('USDT','')
                
                if  'volume' not in result:
                    result = {}
                    result['volume'] = []
                    result['ticks'] = []
                    result['open'] = []
                    result['low'] = []
                    result['high'] = []
                    result['close'] = []
                    result['cost'] = []
                result['volume'].append(float(rate['fundingRate']))
                result['ticks'].append(float(rate['fundingRate']))
                result['open'].append(float(rate['fundingRate']))
                result['low'].append(float(rate['fundingRate']))
                result['high'].append(float(rate['fundingRate']))
                result['close'].append(float(rate['fundingRate']))
                result['cost'].append(float(rate['fundingRate']))

            #response = requests.get(self.base_endpoint + self.candlestick_endpoint.format(int(end),instrument,interval,int(start))).json()["result"]
            df = pd.DataFrame(result)
            #df.to_csv("./hist_data/deribit/btc.txt")
            return df
        except Exception as e:
            print(e)

    def get_start_end(self, num_days):
        ticks_back = num_days * 60 * 24
        end = int(time.time()) * 1000
        start = int(time.time()) * 1000 - 1000 * 60 * ticks_back
        return start, end


    def get_file_symbol(self, sym_full):
        stripped = sym_full.split(".")[0][:3]
        return stripped

    def load_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/history"))
        df_dict = {}
        for sym in histFiles:
            frame = pd.DataFrame().from_csv("./hist_data/history/" +sym)
            df_dict[sym] = frame
        return df_dict
    
    def load_live_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/history"))
        df_dict = {}
        for sym in histFiles:
            frame = pd.DataFrame().from_csv("./hist_data/history/" +sym)
            df_dict[sym] = frame
        return df_dict

    def get_train_frames(self, restrict_val = 0, feature_columns = ['hl_spread', 'oc_spread', 'volume_feature', 'roc_55', 'roc_21', 'roc_8']):
        df_dict = self.load_hist_files()
        coin_and_hist_index = 0
        file_lens = []
        currentHists = {}
        hist_shaped = {}
        coin_dict = {}
        vollist = []
        prefixes = []
        for y in df_dict:
            df = df_dict[y]
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        hist_full_size = mode_len
        vollist = []
        prefixes = []
        for x in df_dict:
            df = df_dict[x]
            col_prefix = self.get_file_symbol(x)
            #as_array = np.array(df)
            if(len(df) == mode_len):
                #print(as_array)
                prefixes.append(col_prefix)
                currentHists[col_prefix] = df
                print(len(df["volume"]))
                print(col_prefix)
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        for ix in vollist:
            print(prefixes[ix])
            #df['vol'] = (df['vol'] - df['vol'].mean())/(df['vol'].max() - df['vol'].min())
            df = currentHists[prefixes[ix]][feature_columns].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            hist_shaped[coin_and_hist_index] = as_array
            coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_full_size

