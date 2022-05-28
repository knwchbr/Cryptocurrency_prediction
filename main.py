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

#กำหนดการ split data
end_train = '2020-05-01'

st.title("Prediction App")
#การเลือกข้อมูล
stocks = ("BTC","ETH","ADA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)
#การโหลดข้อมูล
@st.cache
def load_data(ticker) :
    data = pd.read_csv(f'CoinData\{ticker}-USD.csv')
    dataline = data
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S')
    data = data.loc[:, ['Date', 'Open', 'Close', 'High', 'Low']]
    data = data.set_index('Date')
    data = data.asfreq('D')
    data = data.sort_index()

    df_rw = {'Open': [],'Close': [],'pred_close':[]}
    df_rw = pd.DataFrame(df_rw)
    df_rw['Open'] = data[['Open']].copy()
    df_rw['Close'] = data[['Close']].copy()
    df_rw['pred_close'] = df_rw['Close'].shift(1)
    df_rw.drop(df_rw.index[0], inplace=True)
    y_true = df_rw.loc[end_train:, 'Close']
    y_pred = df_rw.loc[end_train:, 'pred_close']
    metric = mean_absolute_error(y_true, y_pred)

    print(f'Test error: {metric}')
    
    return data, df_rw, dataline

data_load_state = st.text("Load data...")
data,df_rw,dataline = load_data(selected_stock)
data_load_state.text("Loading data...done")

st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data() :
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[:], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

def Create_Model() :
    forecaster = ForecasterAutoreg(
                    regressor = LGBMRegressor(random_state=123),
                    lags      = 12
                    )

    return forecaster

def Predict_Model(data, df_rw, forecaster) :

    print(len(data.loc[:end_train,'Close']))
    print(len(df_rw['Close']))

    # Backtest test data, 1 step
    metric, predictions = backtesting_forecaster(
                                    forecaster = forecaster,
                                    y          = data['Close'],
                                    initial_train_size = len(data.loc[:end_train, 'Close']),
                                    fixed_train_size   = True,
                                    steps      = 1,
                                    refit      = True,
                                    metric     = 'mean_absolute_error',
                                    verbose    = False
                                    )

    return metric, predictions

forecaster = Create_Model()
metric, predictions = Predict_Model(data, df_rw, forecaster)

fig1 = go.Figure(data=[go.Scatter(x=data.index[:],
                       y = data['Close'],
                       name='Train')])
fig1.add_trace(go.Scatter(x=predictions.index[:], y=predictions['pred'], name='Test'))
fig1.layout.update(title_text="Training And Testing Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.subheader(f'Backtest error custom metric : {metric}')

forecaster.fit(y=predictions['pred'])

Time = []
prediction = []

Time = pd.Series(Time)
prediction = pd.Series(prediction)


steps = 30
pred = forecaster.predict(steps=steps)

New = []
for i in range(0,steps) :
  New.append((datetime.strptime("2022-04-30", "%Y-%m-%d") + timedelta(days=i+1)).date())

New = pd.Series(New)
Time = Time.append(New, ignore_index = True)

prediction = prediction.append(pred, ignore_index = True)

fig1 = go.Figure(data=[go.Scatter(x=data.index[:],
                       y = data['Close'],
                       name='Train')])
fig1.add_trace(go.Scatter(x=predictions.index[:], y=predictions['pred'], name='Test'))
fig1.add_trace(go.Scatter(x=Time, y=prediction, name='Prediction'))
fig1.layout.update(title_text="Prediction Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)


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

s = 0
e = len(data)-2
dfpl = dataline

st.subheader('Support and Resistance Line')
dfpl = dataline
fig2 = go.Figure(data=[go.Scatter(x=dfpl['Date'],
                       y = dfpl['Close'],
                       name='Data')])

def AddLine() :
    c = 0
    while 1 :
        if c > len(sr) -1 :
            break
        fig2.add_shape(type='line', x0=dataline.Date[0], y0=sr[c][1],
                        x1=(datetime.strptime("2022-05-01", "%Y-%m-%d") + timedelta(days=i+1)).date(),
                        y1=sr[c][1],
                        line = dict(color="Green", width = 1)
                        )
        c += 1

AddLine()
fig2.add_trace(go.Scatter(x=predictions.index[:], y=predictions['pred'], name='Test'))
fig2.add_trace(go.Scatter(x=Time, y=prediction, name='Prediction'))
fig2.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)