import pandas as pd
from pickle import TRUE
from pandas import Period
import streamlit as st
from datetime import datetime
from datetime import timedelta

import yfinance as yf
from plotly import graph_objs as go

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

data = pd.read_csv(f'CoinData\ETH-USD.csv')

data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S')

def support(data, l, n1, n2) :
  for i in range(l-n1+1, l+1) :
    if data.Low[i] > data.Low[i - 1] :
      return 0
  
  for i in range(l+1, l+n2 + 1) :
    if data.Low[i] < data.Low[i - 1] :
      return 0
  return 1

def resistance(data, l, n1, n2) :
  for i in range(l-n1+1, l+1) :
    if data.High[i] < data.High[i - 1] :
      return 0
  
  for i in range(l+1, l+n2 + 1) :
    if data.High[i] > data.High[i - 1] :
      return 0
  return 1

sr = []
n1 = 3
n2 = 4

for row in range(n1, len(data)-2) :
  if support(data, row, n1, n2) :
    if data.Low[row] > data['Close'].mean() :
        sr.append((data.index[row],data.Low[row], 1))
  if resistance(data, row, n1, n2) :
    if data.High[row] > data['Close'].mean() :
        sr.append((data.index[row], data.High[row], 2))

print(sr)