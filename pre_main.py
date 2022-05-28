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

START = "2012-01-01"
END = "2022-04-30"

st.title("Prediction App")

stocks = ("BTC","ETH","ADA","SOL")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

@st.cache
def load_data(ticker) :
    data = pd.read_csv(f'CoinData\{ticker}-USD.csv')
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done")

st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data() :
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

forecaster = ForecasterAutoreg(
                regressor = LGBMRegressor(random_state=123),
                lags = 12
                )

forecaster.fit(y=data['Close'])

Time = []
prediction = []

Time = pd.Series(Time)
prediction = pd.Series(prediction)


steps = 45
predictions = forecaster.predict(steps=steps)

New = []
for i in range(0,steps) :
  New.append((datetime.strptime(END, "%Y-%m-%d") + timedelta(days=i+1)).date())

New = pd.Series(New)
Time = Time.append(New, ignore_index = True)

prediction = prediction.append(predictions, ignore_index = True)
print(New.head(5))

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
    if data.Low[row] > 15000 :
        sr.append((data.Date[row],data.Low[row], 1))
  if resistance(data, row, n1, n2) :
    if data.High[row] > 15000 :
        sr.append((data.Date[row], data.High[row], 2))

s = 0
e = len(data)-2
dfpl = data

st.subheader('Forecast Data')
dfpl = data
fig1 = go.Figure(data=[go.Scatter(x=dfpl['Date'],
                       y = dfpl['Close'],
                       name='Data')])

fig1.add_trace(go.Scatter(x=Time, y=prediction, name='Prediction'))
fig1.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.subheader('Support and Resistance Line')
dfpl = data
fig2 = go.Figure(data=[go.Scatter(x=dfpl['Date'],
                       y = dfpl['Close'],
                       name='Data')])

def AddLine() :
    c = 0
    while 1 :
        if c > len(sr) -1 :
            break
        fig2.add_shape(type='line', x0=data.Date[0], y0=sr[c][1],
                        x1=(datetime.strptime("2022-05-01", "%Y-%m-%d") + timedelta(days=i+1)).date(),
                        y1=sr[c][1],
                        line = dict(color="Green", width = 1)
                        )
        c += 1

AddLine()
fig2.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)

st.subheader('Test')
dfpl = data
fig3 = go.Figure(data=[go.Scatter(x=dfpl['Date'],
                       y = dfpl['Close'],
                       name='Data')])

def AddLine() :
    c = 0
    while 1 :
        if c > len(sr) -1 :
            break
        fig3.add_shape(type='line', x0=data.Date[0], y0=sr[c][1],
                        x1=(datetime.strptime("2022-05-01", "%Y-%m-%d") + timedelta(days=i+1)).date(),
                        y1=sr[c][1],
                        line = dict(color="Green", width = 1)
                        )
        c += 1

AddLine()
fig3.add_trace(go.Scatter(x=Time, y=prediction, name='Prediction'))
fig3.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig3)