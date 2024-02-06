# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:44:22 2023

@author: spuri1
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:11:26 2023

@author: spuri1
"""
#c('psycopg2','pandas', 'numpy', 'matplotlib', 'seaborn', 'pmdarima', 'sklearn' )


print("Inside RS Plot")
import psycopg2 as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.arima.model import ARIMA
#import statsmodels.api as sm
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
#from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Import Finish")
#Get Data
conn=pg.connect(database=""
              ,user=""
              ,password=""
              ,host=""
              ,port="")
print("DB Connect")
cursor=conn.cursor()
query = "SELECT * FROM sandbox_grwi.sp_module2_test_case1"
cursor.execute(query)
df = pd.DataFrame(cursor.fetchall(), columns=[name[0] for name in cursor.description])
conn.close()
print("Data Fetched")
# Filter Xtandi
df['rx_fill_month'] = pd.to_datetime(df['rx_fill_month'])
df_xtd = df[df['treatment'] == 'XTANDI']
#df_data_xtd = df_data_xtd[df_data_xtd['rx_fill_month'] >= '2020-01-01' ].filter(items=['rx_fill_month', 'total_cnt']).sort_values(by=['rx_fill_month'])
df_xtd = df_xtd.filter(items=['rx_fill_month', 'total_cnt']).sort_values(by=['rx_fill_month'])
df_xtd.set_index('rx_fill_month', inplace = True)

#------------------------------ FORECAST ANOMALY MODEL-------------------------------
# Forcast latest 6 months values
df_xtd_test = df_xtd[-12:]
df_xtd_train = df_xtd[:-12]

# Decompose
#results = seasonal_decompose(df_xtd_train, model='additive')
#results.plot()

# Get differencing order & values
def diff_order(data, max_d = 5):
  np.seterr(divide = 'ignore') 
  for d in range(max_d + 1):
      diff_data = data.diff(d).dropna()      
      adf_result = adfuller(diff_data)
      p_value = adf_result[1]
      if p_value <= 0.05:
         return d, diff_data

d_order, df_xtd_train_diff = diff_order(df_xtd_train)

# Plot acf & pacf for identifying p&q for Arima model    
#sm.graphics.tsa.plot_acf(df_xtd_train_diff,lags=40)
#sm.graphics.tsa.plot_pacf(df_xtd_train_diff,lags=20)

# Used Auto Arima to automaatically identify best model & parameters 
#Run time-series model & predict next 6 months
model = auto_arima(df_xtd_train_diff, max_p=5, max_q=5,d=d_order,
                    m=12,
                    seasonal=True, 
                    suppress_warnings=True, 
                    stepwise=False)
arima = model.fit(df_xtd_train_diff)

# Add predict values
df_xtd['diff_order_predict'] = pd.concat([pd.Series(arima.predict_in_sample()), pd.Series(arima.predict(n_periods = 6))]).astype(int)
df_xtd = df_xtd.apply(lambda x: x.replace('nan',None,regex = True)).fillna(0)

# Inverse differenceing 
df_xtd.reset_index(inplace = True)
for i in range(len(df_xtd)):
  try:
      df_xtd.loc[i,'total_cnt_predict'] = df_xtd.loc[i,'diff_order_predict'] + df_xtd.loc[i-d_order,'total_cnt']
  except KeyError:
     df_xtd.loc[i,'total_cnt_predict'] = df_xtd.loc[i,'diff_order_predict'] 
#df_data_xtd_diff[['total_cnt','predict']].plot() 

# Add 6 months rolling mean & stdv
df_xtd['error'] = df_xtd['total_cnt'] - df_xtd['total_cnt_predict']
df_xtd['mean'] = df_xtd['total_cnt_predict'].rolling(3).mean()
df_xtd['stdv'] = df_xtd['total_cnt_predict'].rolling(3).std()
df_xtd['-s'] = df_xtd['mean'] - (df_xtd['stdv'])
df_xtd['s']  = df_xtd['mean'] + (df_xtd['stdv']) 
df_xtd['-1.5s'] = df_xtd['mean'] - (1.5 * df_xtd['stdv'])
df_xtd['1.5s']  = df_xtd['mean'] + (1.5 * df_xtd['stdv'])
df_xtd['-2s'] = df_xtd['mean'] - (2 * df_xtd['stdv'])
df_xtd['2s']  = df_xtd['mean'] + (2 * df_xtd['stdv'])


#df_xtd.drop(['impact'], axis=1, inplace = True)
# Check anomaly with in confidence intervals
intervals = ['s', '1.5s', '2s']
df_xtd['anomaly_check'] = 'fail'
# Iterate through each row
for i in range(len(df_xtd)):
  for interval in intervals:
      if (df_xtd['total_cnt'].iloc[i] <= df_xtd[interval].iloc[i]) & (df_xtd['total_cnt'].iloc[i] >= df_xtd['-'+interval].iloc[i]):
          df_xtd.loc[i,'anomaly_check'] = interval
          break  # Exit the loop if condition is met
  # If none of the intervals satisfy the condition, 'output' remains 'anomaly'

impact = {'s': 'low', '1.5s': 'medium', '2s': 'high', 'fail': 'anomaly'}
df_xtd['impact'] =  df_xtd['anomaly_check'].map(impact)

#Plot
df_xtd_plt = df_xtd[df_xtd['rx_fill_month'] >= '2020-01-01']
df_xtd_plt.set_index('rx_fill_month', inplace = True)
plt.figure(figsize= (10,6))
#sns.set(style="darkgrid") 
latest_month = df_xtd_plt.index[-1].strftime("%b '%y")
#latest_count = df_xtd_plt['total_cnt'].iloc[-1]
plt.text(df_xtd_plt.index[-1], df_xtd_plt['total_cnt'].iloc[-1], f"{latest_month}", color='red', ha = 'left', va = 'top')
plt.plot(df_xtd_plt['total_cnt'], label = 'ACTUALS', color = 'red' )
plt.plot(df_xtd_plt['total_cnt_predict'], label = 'EXPECTED', color = 'orange')
plt.fill_between(df_xtd_plt.index, df_xtd_plt['-2s'], df_xtd_plt['2s'], linestyle = '--', color = 'gray', alpha = 0.3, label = '-2 STDV. TO 2 STDV')
high_impact = df_xtd_plt[df_xtd_plt['impact'] == 'anomaly']
plt.scatter(high_impact.index, high_impact['total_cnt'], color = '#333F4B', marker = '^', label = 'ANOMALY')
plt.xlabel('MONTHS', fontsize = 12)
plt.ylabel('COUNTS', fontsize = 12)
plt.title('TOTAL CLAIMS (XTANDI)',fontsize = 16)
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol= 5) 
plt.savefig('rs_plot1.png', bbox_inches='tight')
print("Figure Saved")
plt.show()
