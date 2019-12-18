from flask import Flask
import time as t
import pandas as pd
import flask
import matplotlib.pyplot as plt
import numpy as np
import io
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pytz import timezone
from flask_cors import CORS, cross_origin
from keras import backend as K


def twomin():
  K.clear_session()
  data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=BTCUSD&interval=1min&apikey=XZUST95N0Z2S1L6Q')


  data=data.json()
  data=data['Time Series (1min)']
  #print(data)

  kb='Asia/Kabul'
  df= pd.DataFrame(columns=['date','open','high','low','close','volume'])

  for d,p in data.items():
      date=datetime.strptime(d,'%Y-%m-%d %H:%M:%S')
      data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['5. volume'])]
      df.loc[-1,:]=data_row
      df.index=df.index+1   #increasing index everytime a data_row is appended

  df['unix'] = pd.DatetimeIndex ( df['date'] ).astype ( np.int64 )/1000000

  #basic prediction andaza
  data=df.sort_values('date')
  data['close']=data['close'].astype(float)
  data['2min']=np.round(data['close'].rolling(window=3).mean(),2)
  data['2min'][0]=data['2min'][2];
  data['2min'][1]=data['2min'][0];
  lst1=[]
  i=0
  for i in range(0,100):
      d={}
      d.update({"x":data['unix'].iloc[i]})
      d.update({"y":data['close'].iloc[i]})
      d.update({"z":data['2min'].iloc[i]})	    
      #print(d)
      lst1.append(d)

  #lst1=str(lst1)

  x =df.date[99]
  datetime_obj_utc = x.replace(tzinfo=timezone('US/Eastern'))
  format = "%Y-%m-%d %H:%M:%S"

  india = datetime_obj_utc.astimezone(timezone(kb))
  print(india)
  india = india + timedelta(minutes=5)
  r = str(india.strftime(format))
  #r = r + timedelta(hours=2)

  closing_value = df.close[99]

  s = str(closing_value)


  data=df.sort_values('date')
  data['real']=data['close'].astype(float)
  data['predicted']=np.round(data['close'].rolling(window=3).mean(),2)

  trend='UpTrend'
  if (data['predicted'][99]<=data['real'][99]):
      trend='UpTrend'
      #print(" Prediction: UpTrend " , end=" ")
  else:
      trend='Downtrend'
      #print(" Prediction: DownTrend " , end=" ")

  dif1=df['open'][99]-df['open'][98]
  dif2=df['open'][99]-df['open'][98]
  dif=abs((dif1+dif2)/2);

  if(trend=='UpTrend'):
      a=str(int(data['real'][99]+dif)) + " 1"

  else:
      a=str(int(data['real'][99]-dif)) + " 0"
	
  ist="IST"
  lst=[]
  lst.append(r)
  lst.append(ist)
  lst.append(s)
  lst.append(a)
  lst.append(lst1)
  
  return lst

def oneday():
	K.clear_session()
	data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=BTCUSD&interval=1min&apikey=45PX1LNA4QNXA2Q9')
	data=data.json()

	l = list(data['Time Series (1min)'])
	l.sort()

	# get market info for bitcoin from the start of 2013 to the current day
	bitcoin_market_info1 = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+t.strftime("%Y%m%d"))[0]

	temp = {'Date' : t.strftime("%b %d, %Y") , 'Open*' : float(data['Time Series (1min)'][l[-1]]['1. open']), 'High' : float(data['Time Series (1min)'][l[-1]]['2. high']), 'Low' : float(data['Time Series (1min)'][l[-1]]['3. low']) , 'Close**' : float(data['Time Series (1min)'][l[-1]]['4. close']), 'Volume' : bitcoin_market_info1['Volume'][0] , 'Market Cap' :  bitcoin_market_info1['Market Cap'][0] }




	bitcoin_market_info1=pd.concat([pd.DataFrame([temp]),bitcoin_market_info1],ignore_index=True,sort=False)
	bitcoin_market_info1=bitcoin_market_info1[['Date','Open*','High','Low','Close**','Volume','Market Cap']]

	# convert the date string to the correct date format
	bitcoin_market_info1 = bitcoin_market_info1.assign(Date=pd.to_datetime(bitcoin_market_info1['Date']))

	# when Volume is equal to '-' convert it to 0
	bitcoin_market_info1.loc[bitcoin_market_info1['Volume']=="-",'Volume']=0

	# convert to int
	bitcoin_market_info1['Volume'] = bitcoin_market_info1['Volume'].astype('int64')

	# sometime after publication of the blog, coinmarketcap starting returning asterisks in the column names
	# this will remove those asterisks
	bitcoin_market_info1.columns = bitcoin_market_info1.columns.str.replace("*", "")
	bitcoin_market_info = bitcoin_market_info1.drop("Market Cap",1)
	# look at the first few rows
	#bitcoin_market_info.head()

	# get market info for ethereum from the start of 2015 to the current day
	eth_market_info1 = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+t.strftime("%Y%m%d"))[0]

	data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ETHUSD&interval=1min&apikey=45PX1LNA4QNXA2Q9')
	data=data.json()

	l = list(data['Time Series (1min)'])
	l.sort()
	data['Time Series (1min)'][l[-1]]

	temp = {'Date' : t.strftime("%b %d, %Y") , 'Open*' : float(data['Time Series (1min)'][l[-1]]['1. open']), 'High' : float(data['Time Series (1min)'][l[-1]]['2. high']), 'Low' : float(data['Time Series (1min)'][l[-1]]['3. low']) , 'Close**' : float(data['Time Series (1min)'][l[-1]]['4. close']), 'Volume' : eth_market_info1['Volume'][0] , 'Market Cap' :  eth_market_info1['Market Cap'][0] }

	eth_market_info1=pd.concat([pd.DataFrame([temp]),eth_market_info1],ignore_index=True,sort=False)
	eth_market_info1=eth_market_info1[['Date','Open*','High','Low','Close**','Volume','Market Cap']]

	# convert the date string to the correct date format
	eth_market_info1 = eth_market_info1.assign(Date=pd.to_datetime(eth_market_info1['Date']))

	# sometime after publication of the blog, coinmarketcap starting returning asterisks in the column names
	# this will remove those asterisks
	eth_market_info1.columns = eth_market_info1.columns.str.replace("*", "")

	# look at the first few rows
	eth_market_info=eth_market_info1.drop('Market Cap',1)

	#eth_market_info.head()

	#appending bt and eth to column names
	bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
	eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]

	#merge two tabels
	market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])

	market_info = market_info[market_info['Date']>='2016-01-01']

	for coins in ['bt_', 'eth_']: 
		  kwargs = { coins+'day_diff': lambda x: ((x[coins+'Close'])-(x[coins+'Open']))/x[coins+'Open']}
		  market_info = market_info.assign(**kwargs)
	#market_info.head()

	split_date = '2018-12-01'

	for coins in ['bt_', 'eth_']: 
		  kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
		    coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
		  market_info = market_info.assign(**kwargs)

	model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_'] 
		                           for metric in ['Close','Volume','close_off_high','volatility']]]
	# need to reverse the data frame so that subsequent rows represent later timepoints
	model_data = model_data.sort_values(by='Date')
	#model_data.head()

	#this is important library
	from datetime import date,timedelta


	# we don't need the date columns anymore
	test_set1 = model_data[model_data['Date']>=pd.Timestamp(date.today()-timedelta(days=11))]
	test_set1 = test_set1.drop('Date', 1)

	window_len = 10
	norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['Close','Volume']]

	#LSTM_training_inputs[0]

	LSTM_test_inputs1 = []
	for i in range(len(test_set1)-window_len):
		  temp_set1 = test_set1[i:(i+window_len)].copy()
		  for col in norm_cols:
		      temp_set1.loc[:, col] = temp_set1[col]/temp_set1[col].iloc[0] - 1
		  LSTM_test_inputs1.append(temp_set1)

	#LSTM_test_inputs1

	# I find it easier to work with numpy arrays rather than pandas dataframes


	# I find it easier to work with numpy arrays rather than pandas dataframes
	# especially as we now only have numerical data


	LSTM_test_inputs1 = [np.array(LSTM_test_input) for LSTM_test_input in LSTM_test_inputs1]
	LSTM_test_inputs1 = np.array(LSTM_test_inputs1)
	#LSTM_test_inputs1.shape

	from keras.models import load_model
	# load model
	bt_model = load_model('bt_model.h5')
	#Current price 
	cur=str(test_set1['bt_Close'][window_len:][0])
	#predicted price for 15 Aug
	val= list(((np.transpose(bt_model.predict(LSTM_test_inputs1))+1) * test_set1['bt_Close'].values[:-window_len][0]))
	val=str(val[0][0])
	bitcoin_market_info1['unix'] = pd.DatetimeIndex ( bitcoin_market_info1['Date'] ).astype ( np.int64 )/1000000


	lst1=[]
	i=0
	for i in range(0,30):
		d={}
		d.update({"x":bitcoin_market_info1['unix'].iloc[i]})
		d.update({"y":bitcoin_market_info1['Close'].iloc[i]})
		#print(d)
		lst1.append(d)

	from datetime import datetime, timedelta
	dt=datetime.today() + timedelta(days=1)
	timestamp = (dt - datetime(1970, 1, 1)).total_seconds()
	dte = int(timestamp*1000)

	d={}
	d.update({"x":dte})
	d.update({"y":val})
	lst1.insert(0,d)

	lst=[]
	tm=t.strftime("%Y-%m-%d %H:%M:%S")
	ist="IST"
	if val > cur:
		ud=1
	else:
		ud=0
	#lst.append(t.strftime("%Y-%m-%d %H:%M:%S")
	lst.append(tm)
	lst.append(ist)
	lst.append(cur)
	lst.append(val)
	lst.append(ud)
	lst.append(lst1)
	return lst


def weekpred():
	data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=BTCUSD&interval=1min&apikey=JKVX9HTFNJ6N8QNS')
	data=data.json()

	l = list(data['Time Series (1min)'])
	l.sort()

	# get market info for bitcoin from the start of 2013 to the current day
	bitcoin_market_info1 = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+t.strftime("%Y%m%d"))[0]

	temp = {'Date' : t.strftime("%b %d, %Y") , 'Open*' : float(data['Time Series (1min)'][l[-1]]['1. open']), 'High' : float(data['Time Series (1min)'][l[-1]]['2. high']), 'Low' : float(data['Time Series (1min)'][l[-1]]['3. low']) , 'Close**' : float(data['Time Series (1min)'][l[-1]]['4. close']), 'Volume' : bitcoin_market_info1['Volume'][0] , 'Market Cap' :  bitcoin_market_info1['Market Cap'][0] }




	bitcoin_market_info1=pd.concat([pd.DataFrame([temp]),bitcoin_market_info1],ignore_index=True,sort=False)
	bitcoin_market_info1=bitcoin_market_info1[['Date','Open*','High','Low','Close**','Volume','Market Cap']]

	# convert the date string to the correct date format
	bitcoin_market_info1 = bitcoin_market_info1.assign(Date=pd.to_datetime(bitcoin_market_info1['Date']))

	# when Volume is equal to '-' convert it to 0
	bitcoin_market_info1.loc[bitcoin_market_info1['Volume']=="-",'Volume']=0

	# convert to int
	bitcoin_market_info1['Volume'] = bitcoin_market_info1['Volume'].astype('int64')

	# sometime after publication of the blog, coinmarketcap starting returning asterisks in the column names
	# this will remove those asterisks
	bitcoin_market_info1.columns = bitcoin_market_info1.columns.str.replace("*", "")
	bitcoin_market_info = bitcoin_market_info1.drop("Market Cap",1)
	# look at the first few rows
	#bitcoin_market_info.head()

	# get market info for ethereum from the start of 2015 to the current day
	eth_market_info1 = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+t.strftime("%Y%m%d"))[0]

	data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ETHUSD&interval=1min&apikey=JKVX9HTFNJ6N8QNS')
	data=data.json()

	l = list(data['Time Series (1min)'])
	l.sort()
	data['Time Series (1min)'][l[-1]]

	temp = {'Date' : t.strftime("%b %d, %Y") , 'Open*' : float(data['Time Series (1min)'][l[-1]]['1. open']), 'High' : float(data['Time Series (1min)'][l[-1]]['2. high']), 'Low' : float(data['Time Series (1min)'][l[-1]]['3. low']) , 'Close**' : float(data['Time Series (1min)'][l[-1]]['4. close']), 'Volume' : eth_market_info1['Volume'][0] , 'Market Cap' :  eth_market_info1['Market Cap'][0] }

	eth_market_info1=pd.concat([pd.DataFrame([temp]),eth_market_info1],ignore_index=True,sort=False)
	eth_market_info1=eth_market_info1[['Date','Open*','High','Low','Close**','Volume','Market Cap']]

	# convert the date string to the correct date format
	eth_market_info1 = eth_market_info1.assign(Date=pd.to_datetime(eth_market_info1['Date']))

	# sometime after publication of the blog, coinmarketcap starting returning asterisks in the column names
	# this will remove those asterisks
	eth_market_info1.columns = eth_market_info1.columns.str.replace("*", "")

	# look at the first few rows
	eth_market_info=eth_market_info1.drop('Market Cap',1)

	#eth_market_info.head()

	#appending bt and eth to column names
	bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
	eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]

	#merge two tabels
	market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])

	market_info = market_info[market_info['Date']>='2016-01-01']

	for coins in ['bt_', 'eth_']: 
		  kwargs = { coins+'day_diff': lambda x: ((x[coins+'Close'])-(x[coins+'Open']))/x[coins+'Open']}
		  market_info = market_info.assign(**kwargs)
	#market_info.head()

	split_date = '2018-12-01'

	for coins in ['bt_', 'eth_']: 
		  kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
		    coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
		  market_info = market_info.assign(**kwargs)

	model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_'] 
		                           for metric in ['Close','Volume','close_off_high','volatility']]]
	# need to reverse the data frame so that subsequent rows represent later timepoints
	model_data = model_data.sort_values(by='Date')
	#model_data.head()

	#this is important library
	from datetime import date,timedelta


	# we don't need the date columns anymore
	#test_set1 = model_data[model_data['Date']>=pd.Timestamp(date.today()-timedelta(days=11))]
	#test_set1 = test_set1.drop('Date', 1)
	# we don't need the date columns anymore
	training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]

	test_set1 = model_data[model_data['Date']>=pd.Timestamp(date.today()-timedelta(days=11))]

	training_set = training_set.drop('Date', 1)

	test_set = test_set.drop('Date', 1)

	test_set1 = test_set1.drop('Date', 1)

	window_len = 10
	norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['Close','Volume']]

	LSTM_test_inputs = []
	for i in range(len(test_set)-window_len):
		temp_set = test_set[i:(i+window_len)].copy()
		for col in norm_cols:
		    temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
		LSTM_test_inputs.append(temp_set)


	LSTM_test_outputs = (test_set['eth_Close'][window_len:].values/test_set['eth_Close'][:-window_len].values)-1

	#LSTM_training_inputs[0]

	LSTM_test_inputs1 = []
	for i in range(len(test_set1)-window_len):
		  temp_set1 = test_set1[i:(i+window_len)].copy()
		  for col in norm_cols:
		      temp_set1.loc[:, col] = temp_set1[col]/temp_set1[col].iloc[0] - 1
		  LSTM_test_inputs1.append(temp_set1)

	#LSTM_test_inputs1

	# I find it easier to work with numpy arrays rather than pandas dataframes


	# I find it easier to work with numpy arrays rather than pandas dataframes
	# especially as we now only have numerical data

	# I find it easier to work with numpy arrays rather than pandas dataframes
	# especially as we now only have numerical data
	#LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
	#LSTM_training_inputs = np.array(LSTM_training_inputs)

	LSTM_test_inputs = [np.array(LSTM_test_input) for LSTM_test_input in LSTM_test_inputs]
	LSTM_test_inputs = np.array(LSTM_test_inputs)
	#LSTM_test_inputs1.shape

	from keras.models import load_model
	# load model
	bt_model = load_model('bt_model7.h5')

	pred_range = 7

	bt_pred_prices = ((bt_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)*\
		                 test_set['bt_Close'].values[:-(window_len + pred_range)][::7].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))

	cur=str(test_set1['bt_Close'][window_len:][0])


	val=str(bt_pred_prices[37][6])
	bitcoin_market_info1['unix'] = pd.DatetimeIndex ( bitcoin_market_info1['Date'] ).astype ( np.int64 )/1000000


	# convert the date string to the correct date format
	#bitcoin_market_info1 = bitcoin_market_info1.assign(Date=pd.to_datetime(bitcoin_market_info1['Date']))
	bitcoin_market_info1

	lst1=[]
	i=0
	for i in range(0,84):
		d={}
		if i%6==0:
		  d.update({"x":bitcoin_market_info1['unix'].iloc[i]})
		  d.update({"y":bitcoin_market_info1['Close'].iloc[i]})
		  #print(d)
		  lst1.append(d)

	from datetime import datetime, timedelta
	dt=datetime.today() + timedelta(days=7)
	timestamp = (dt - datetime(1970, 1, 1)).total_seconds()
	dte = int(timestamp*1000)

	d={}
	d.update({"x":dte})
	d.update({"y":val})
	lst1.insert(0,d) 


	lst=[]
	tm=t.strftime("%Y-%m-%d %H:%M:%S")
	ist="IST"
	if val > cur:
		ud=1
	else:
		ud=0
	#lst.append(t.strftime("%Y-%m-%d %H:%M:%S")
	lst.append(tm)
	lst.append(ist)
	lst.append(cur)
	lst.append(val)
	lst.append(ud)
	lst.append(lst1)
	return lst


from flask import Flask, send_file, make_response
app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)
#api = Api(app)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response



@app.route('/', methods=['GET'])
def data_matrix():
  K.clear_session()
  return flask.jsonify(twomin())
  K.clear_session()

#time and zone               
@app.route('/1', methods=['GET'])
def dat_matrix():
  K.clear_session()
  return flask.jsonify(oneday())
  K.clear_session()

#dynamic graph value
@app.route('/2', methods=['GET'])
def real_matrix():
  K.clear_session()
  return flask.jsonify(weekpred())
  K.clear_session()

if __name__ == '__main__':
	app.run(host = '0.0.0.0',port=5000)

