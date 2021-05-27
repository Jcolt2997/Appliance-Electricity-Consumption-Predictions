import numpy as np
import pandas as pd
from numpy import linalg as LA
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import pmdarima as pmd
from pmdarima.metrics import smape
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2
from scipy import signal
from pandas import DataFrame
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)

# read the data
df = pd.read_csv('KAG_energydata_complete.csv')
df['Date']=pd.to_datetime(df['date'])
del df['date']
df.set_index('Date', inplace = True)


# 5.a: plot the Appliance vs time
def difference(dataset, interval):
   diff = []
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return diff


# applied 144 differencing as
# there were 10 minute intervals in the data and 6 intervals/hr * 24 hr/day = 144
Diff_1 = difference(df.Appliances, 144)
DF = pd.DataFrame(Diff_1, index=df.index[ 144:])
DF.rename(columns={0:'Appliances'}, inplace=True)


plt.figure()
plt.subplot(211)
plt.plot(df.index, df.Appliances)
plt.xlabel('Date')
plt.xticks(df.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.title('Appliance Electricity (Wh) Over Time')
plt.subplot(212)
plt.plot(DF.index, DF.Appliances)
plt.xlabel('Date')
plt.xticks(DF.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.title('Differenced Electricity (Wh) Over Time')
plt.tight_layout()
plt.show()

# =============== #
# Rolling Average #
# =============== #
def rolling_mv(y, title):
    means = []
    vars = []

    for i in range(1, len(y)):
        means.append(y[:i].mean())
        vars.append(y[:i].var())

    # plot rolling mean
    plt.figure()
    plt.subplot(211)
    plt.plot(means, label= 'Mean')
    plt.title("Rolling Mean "  + str(title))
    plt.xlabel('Time')
    plt.ylabel("Rolling Mean")
    plt.subplot(212)
    plt.plot(vars, label= 'Variance')
    plt.title("Rolling Variance " + str(title))
    plt.xlabel('Time')
    plt.ylabel("Rolling Variance")
    plt.tight_layout()
    plt.show()

# plot rolling average
rolling_mv(df.Appliances, 'of Original Data')
rolling_mv(DF.Appliances, 'of Differenced Data')


 # 5.b: ACF/PACF plots ACF Plot
def autocorr(x,lag):
     l = range(lag+1)
     x_br = np.mean(x)
     autocorr = []
     for i in l:
         num = 0
         var = 0
         for j in range(i, len(x)):
             num += np.sum(x[j]- x_br) *(x[j-i] -x_br)
         var = np.sum((x - x_br) ** 2)
         autocorr.append(num/var)
     return autocorr

def ACF_Plot(x,lag):
    lg = np.arange(-lag, lag + 1)
    x = x[0:lag + 1]
    rx = x[::-1]
    rxx = rx[:-1] + x
    plt.stem(lg, rxx)



# ACF plot of raw data
plt.figure()
bound = 1.96/np.sqrt(len(df.Appliances))
ACF_Plot(autocorr(df.Appliances,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# ACF plot of differenced data
plt.figure()
bound = 1.96/np.sqrt(len(DF.Appliances))
ACF_Plot(autocorr(DF.Appliances,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Differenced Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


def PACF_ACF_Plot(x,lags,title):
    plt.figure()
    plt.subplot(211)
    plt.xlabel('Lags')
    plt.ylabel('ACF Value')
    plt.title('ACF and PACF plot ' + str(title))
    sm.graphics.tsa.plot_acf(x, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plt.xlabel('Lags')
    plt.ylabel('PACF Value')
    sm.graphics.tsa.plot_pacf(x, ax=plt.gca(), lags=lags)
    plt.tight_layout()
    plt.show()

PACF_ACF_Plot(df.Appliances,500, 'of Raw Data')
PACF_ACF_Plot(DF.Appliances,500, 'of Differenced Data')

 # 5c: Matrix Correlation
# Q2: Heat Map
plt.figure()
corr_t= df.corr()
# create correlation coefficient heatmap for x,y,g, and z
ax = sns.heatmap(corr_t, vmin=-1, vmax= 1, center=0, cmap = sns.diverging_palette(20,220, n=200), square= True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment= 'right')
plt.title('Correlation Plot')
plt.tight_layout()
plt.show()
 # 5d:Cleaning

# tested other forms of differencing


 # 5e:Split data of differenced data

df2= df[[ 'Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint', 'rv1', 'rv2']]
# df.set_index('Date')


df_y= df2[['Appliances']]

df_x = df2[['T1', 'lights','RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint', 'rv1', 'rv2']]


# x data
df_x_t, df_x_f= train_test_split(df_x, shuffle= False, test_size=0.2)
# differenced
df_y_t, df_y_f = train_test_split(DF, shuffle= False, test_size=0.2)
# undifferenced
df_y_t_n, df_y_f_n = train_test_split(df_y, shuffle= False, test_size=0.2)
#first date 2016-01-11 17:00:00
#last date  2016-05-27 18:00:00

 #  6: Stationary
def ADF_Cal(x):
    result = adfuller(x)

    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# Data appeared to be stationary with p-value of 0.00000
ADF_Cal(df_y_t.Appliances) #seasonal data
ADF_Cal(DF.Appliances) # non-seasonal

# 7: Time series decomposition

Appliances = df['Appliances']
Appliances = pd.Series(np.array(df['Appliances']),
                      index = pd.date_range('2016-01-11 17:00:00', periods= len(Appliances)), name= 'Electricity (Wh)')

res = STL(Appliances).fit()
fig = res.plot()
plt.ylabel('Residual')
plt.xlabel('Iterations')
plt.tight_layout()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

R = np.array(R)
S = np.array(S)
T = np.array(T)

# strength of seasonality and Trend
Ft = np.max([0,1 - np.var(R)/np.var(T+R)])
Fs = np.max([0,1 - np.var(R)/np.var(S+R)])

print('The strength of trend for this data set is', Ft.round(4))

print('The strength of seasonality for this data set is ', Fs.round(4))

plt.figure()
plt.plot(df.index,T, label= 'Trend')
plt.plot(df.index,R, label= 'Residual')
plt.plot(df.index,S, label= 'Seasonal')
plt.title('Trend, Residual, and Seasonal Plot')
plt.xticks(df.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

adjusted_seasonal = Appliances - S # Adjusted Seasonal Dataset
adjusted_seasonal.to_frame(name = 'Val')
detrended = Appliances - T # detrended data
detrended.to_frame(name = 'Val')


R = np.array(R)
A_S = np.array(adjusted_seasonal)
D_T = np.array(detrended)



plt.figure()
plt.plot(df.index,Appliances, label= 'Original Data', color = 'black')
plt.plot(df.index,adjusted_seasonal.values, label= 'Adjusted Seasonal', color = 'yellow')
plt.xticks(df.index[::4500], fontsize= 10)
plt.title('Seasonaly Adjusted Data vs. Differenced')
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df.index,Appliances, label= 'Original Data')
plt.plot(df.index,detrended.values, label= 'Detrended')
plt.xticks(df.index[::4500], fontsize= 10)
plt.title('Detrended Data vs. Original')
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

# 8: Holt-Winter method

# use training data to fit model
model = ets.ExponentialSmoothing(df_y_t_n['Appliances'], damped_trend= True, seasonal_periods=288, trend='add', seasonal='add').fit()

# prediction on train set
df_y_t_HW = model.forecast(steps=len(df_y_t_n['Appliances']))
df_y_t_HW = pd.DataFrame(df_y_t_HW, columns=['Appliances']).set_index(df_y_t_n.index)
# made prediction on test set
df_y_f_HW = model.forecast(steps=len(df_y_f_n['Appliances']))
df_y_f_HW = pd.DataFrame(df_y_f_HW, columns=['Appliances']).set_index(df_y_f_n.index)



# print the summary
print(model.summary())

# model assessment
def mse(errors):
   return  np.sum(np.power(errors,2))/len(errors)
# tain data
df_y_t_HW_error = np.array(df_y_t_n['Appliances'] - df_y_t_HW['Appliances'])
print("Mean square error for the Holt-Winter method prediction on Electricity (Wh) is ", mse(df_y_t_HW_error).round(4))
print(sm.stats.acorr_ljungbox(df_y_t_HW_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 23100.165439 with a p-value of 0.0')
print('the mean of the Holt-winter model prediction error is', np.mean(df_y_t_HW_error))
print('the variance of the Holt-winter model prediction error is', np.var(df_y_t_HW_error))
print('the RMSE of the Holt-winter model prediction error is, ', mean_squared_error(df_y_t_n['Appliances'], df_y_t_HW['Appliances'], squared=False))

# test data
df_y_f_HW_error = np.array(df_y_f_n['Appliances'] - df_y_f_HW['Appliances'])
print("Mean square error for the Holt-Winter method forecasting on Electricity (Wh) is ", mse(df_y_f_HW_error).round(4))
print(sm.stats.acorr_ljungbox(df_y_f_HW_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 5138.065419 with a p-value of 0.0')
print('the mean of the Holt-winter model error is', np.mean(df_y_f_HW_error))
print('the variance of the Holt-winter model error is', np.var(df_y_f_HW_error))
print('the RMSE of the Holt-winter model error is, ', mean_squared_error(df_y_f_n['Appliances'], df_y_f_HW['Appliances'], squared=False))
print('the variance of the prediction error appeared larger than the variance of the testing error')


# plot Holt-Winter model

# plot of full model
plt.figure()
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Holt-Winter Method on Data')
plt.plot(df_y_t_n.index,df_y_t_n.Appliances,label= "Train Data", color = 'blue')
plt.plot(df_y_f_n.index,df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_HW.set_index(df_y_f_HW.index), label = 'Forecasting Data', color = 'yellow')
plt.xticks(df_y.index[::4500], fontsize= 10)
plt.legend()
plt.tight_layout()
plt.show()


# plot of test data
plt.figure()
plt.plot(df_y_f_n.index,df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_HW.set_index(df_y_f_HW.index), label = 'Forecasting Data', color = 'yellow')
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title(f'Holt-Winter Method on Data with MSE = {mse(df_y_f_HW_error).round(4)}')
plt.xticks(df_y_f_n.index[::725], fontsize= 10)
plt.legend()
plt.tight_layout()
plt.show()

# note
# mse 87444 # s 288

# holt-winter train data
plt.figure()
m_pred_f = 1.96/np.sqrt(len(df_y_t_n.Appliances))
ACF_Plot(autocorr(df_y_t_HW_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('Holt-Winter Train Error ACF Plot with 90 Lags')
plt.axhspan(-m_pred_f,m_pred_f,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


# holt winter test data
plt.figure()
m_pred_f = 1.96/np.sqrt(len(df_y_f_n.Appliances))
ACF_Plot(autocorr(df_y_f_HW_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('Holt-Winter Test Error ACF Plot with 90 Lags')
plt.axhspan(-m_pred_f,m_pred_f,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# 9: Feature Selection collinearity

X = df_x_t[['T1', 'lights','RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint', 'rv1', 'rv2']].values
Y = df_y_t_n.values
X = sm.add_constant(X)
print("this is X dim", X.shape)
H = np.matmul(X.T,X)
print('This is H dim', H.shape)
s,d,v = np.linalg.svd(H)
print('SingularValues = ', d)
#Condition number
print(" the condition number for X is = ", LA.cond(X))
# the condition number for X is =  1.5823924406110067e+17

# Q:10 model building

df_x_t = sm.add_constant(df_x_t)
df_x_f = sm.add_constant(df_x_f)

# full model
model = sm.OLS(df_y_t_n,df_x_t)
results = model.fit()
print(results.summary())

#removed 'rv1 and rv2' as it had p-value of 0.700
b_1= df_x_t[['const','T1', 'lights','RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_1)
results = model.fit()
print(results.summary())



#removed 'const' as it had p-value of 0.553
b_2= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_2)
results = model.fit()
print(results.summary())

#removed 'T7' as they had p-value of 0.426
b_3= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'RH_7', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_3)
results = model.fit()
print(results.summary())

#removed 'RH_7' as they had p-value of 0.415
b_4= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_4)
results = model.fit()
print(results.summary())

#removed 'RH_5' as they had p-value of 0.308
b_6= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T5', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_6)
results = model.fit()
print(results.summary())

#removed 'T5' as they had p-value of 0.366
b_7= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_7)
results = model.fit()
print(results.summary())

#removed 'RH_out' as they had p-value of 0.109
b_8= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'Windspeed',
        'Visibility', 'Tdewpoint']]
model = sm.OLS(df_y_t_n,b_8)
results = model.fit()
print(results.summary())

#removed 'Tdewpoint' as they had p-value of 0.153
b_9= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'Windspeed',
        'Visibility']]
model = sm.OLS(df_y_t_n,b_9)
results = model.fit()
print(results.summary())

#removed 'RH_9' as they had p-value of 0.050
b_10= df_x_t[['lights','T1', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'T_out', 'Press_mm_hg', 'Windspeed',
        'Visibility']]
model = sm.OLS(df_y_t_n,b_10)
results = model.fit()
print(results.summary())

#removed 'T1' as they had standard error of 2.215
b_11= df_x_t[['lights', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T9', 'T_out', 'Press_mm_hg', 'Windspeed',
        'Visibility']]
model = sm.OLS(df_y_t_n,b_11)
results = model.fit()
print(results.summary())



#removed 'Press_mm_hg' as they had p-value of 0.780
# no more multicolliniarity present
b_12= df_x_t[['lights', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T_out', 'Windspeed',
        'Visibility']]
model = sm.OLS(df_y_t_n,b_12)
results_t = model.fit()
print(results_t.summary())


# predictions on backward regression

# #make predictions on training data
ypred = results_t.predict(df_x_t[['lights', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T_out', 'Windspeed',
        'Visibility']])
# prediction data
pred_t = np.array(df_y_t_n['Appliances']) - ypred


# #make predictions on testing data
ypred_f = results_t.predict(df_x_f[['lights', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T_out', 'Windspeed',
        'Visibility']])
# forecasting data
pred_f = np.array(df_y_f_n['Appliances']) - ypred_f


# Q8:
df_x_t= df_x_t[['lights', 'RH_1', 'T2', 'RH_2', 'T3',
        'RH_3', 'T4', 'RH_4', 'T6', 'RH_6', 'T8',
        'RH_8', 'T_out', 'Windspeed',
        'Visibility']].copy()

plt.figure()
plt.plot(df_x_t.index, df_y_t_n.values, label = 'Training set')
plt.xticks(df.index[::4500], fontsize= 10)
# I found it difficult to read the data with the prediction data plotted so I uncommented it.
# if you wish to see the prediction data just uncomment the line below.
# plt.plot(df_x_t.index, pred_t, label = 'Prediction values')
plt.plot(df_x_f.index, df_y_f_n, label = 'Test set')
plt.plot(df_x_f.index,ypred_f, label = 'Forecasted Values')
plt.legend()
plt.tight_layout()
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.title('OLS model Prediction Plot')
plt.show()

plt.figure()
plt.subplot(211)
m_pred_f = 1.96/np.sqrt(len(df_y_t_n.Appliances))
ACF_Plot(autocorr(pred_t.values,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of MLR Prediction Values on Electricity (Wh) with 90 Lags')
plt.axhspan(-m_pred_f,m_pred_f,alpha = .1, color = 'black')
plt.subplot(212)
m_pred_f = 1.96/np.sqrt(len(df_y_f_n.Appliances))
ACF_Plot(autocorr(pred_f.values,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of MLR Forecasting Values on Electricity (Wh) with 90 Lags')
plt.axhspan(-m_pred_f,m_pred_f,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()



# model assessment
def mse(errors):
   return  np.sum(np.power(errors,2))/len(errors)
# train data
print("Mean square error for the regression method prediction on Electricity (Wh) is ", mse(pred_t).round(4))
print(sm.stats.acorr_ljungbox(pred_t, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 12906.653091 with a p-value of  0.0')
print('the mean of the regression model prediction error is', np.mean(pred_t))
print('the variance of the regression model prediction error is', np.var(pred_t))
print('the RMSE of the regression model prediction error is, ', mean_squared_error(df_y_t_n['Appliances'], pred_t.values, squared=False))

# test data
print("Mean square error for the regression method forecasting on Electricity (Wh) is ", mse(pred_f).round(4))
print(sm.stats.acorr_ljungbox(pred_f, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 3890.720017  with a p-value of  0.0')
print('the mean of the regression model forecasting error is', np.mean(pred_f))
print('the variance of the regression model forecasting error is', np.var(pred_f))
print('the RMSE of the regression model forecasting error is, ', mean_squared_error(df_y_f_n['Appliances'], pred_f.values, squared=False))
print('the variance of the prediction error appeared larger than the variance of the testing error')

# Q11: ARMA models

def Autocorr(df, lg):
    mean = np.mean(df)
    m = len(df)
    num = []
    for i in range(lg, m):
        num_value = (df[i] - mean) * (df[i - lg] - mean)
        num.append(num_value)
    num_sum = sum(num)
    den= []
    for j in range(0, m):
        den_val = (df[j] - mean) ** 2
        den.append(den_val)
    den_sum = sum(den)
    result = round(num_sum / den_sum, 4)
    return result


def GPAC_Matrix(ry, j, k):
    col = []
    for i in range(j):
        col.append(ry[i + 1] / ry[i])
    # convert values to dataframe
    table = pd.DataFrame(col, index=np.arange(0, j).tolist())
    # list for values in the GPAC matrix
    val = []
    # for K in GPAC, you do not want to include column 1 as it will all be 1's
    for a in range(2, k + 1):
        for f in range(j):
            b_val = []
            t_val = []
            for d in range(a):
                den = []
                for h in range(a):
                    den.append(ry[abs(f - h + d)])
                b_val.append(den.copy())
                t_val.append(den.copy())
            mes = a
            for l in range(a):
                t_val[l][mes - 1] = ry[l + 1 + f]

            pac = np.linalg.det(t_val) / np.linalg.det(b_val)
            val.append(pac)
    # reshame the GPAC value so there is no 0 row in k
    GPAC = np.array(val).reshape(k -1, j)
    #correctly transpose the data
    GPAC_T = pd.DataFrame(GPAC.T)
    GPAC_F = pd.concat([table, GPAC_T], axis=1)
    GPAC_F.columns = list(range(1, k + 1))
    return GPAC_F


#ACF for 20 lags
acf =[]
for n in range(0,40):
    ac = Autocorr(df_y_t['Appliances'],n)
    acf.append(ac)


# GPAC plot

GPAC = GPAC_Matrix(acf,10,10)
plt.figure()
sns.heatmap(GPAC,  annot=True)
plt.title('GPAC Table')
plt.xlabel('k values')
plt.ylabel('j values')
plt.show()

# select parameters from GPAC ARMA:(3,0),(3,1), and (3,2)

na = 3
nb = 0
# ARMA model
model_1 = sm.tsa.ARMA(df_y_t_n.Appliances,(na, nb)).fit(trend='nc',disp=0)
for i in range(na):
    print("The AR coefficient a{}".format(i), "is:", model_1.params[i])
for i in range(nb):
    print("The MA coefficient a{}".format(i), "is:", model_1.params[i + na])
print(model_1.summary())



#MSE calculation
# train data
arma_1_pred_t = model_1.predict(start=0, end=15787)
arma_1_t_error = df_y_t_n.Appliances - arma_1_pred_t.values

# test data
arma_1_pred_f = model_1.predict(start=15788, end=19734)
arma_1_f_error = df_y_f_n.Appliances - arma_1_pred_f.values



plt.figure()
bound = 1.96/np.sqrt(len(arma_1_t_error))
ACF_Plot(autocorr(arma_1_t_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARMA (3,0) ACF Plot Predicting Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


plt.figure()
bound = 1.96/np.sqrt(len(arma_1_f_error))
ACF_Plot(autocorr(arma_1_f_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARMA (3,0) ACF Plot Forecasting Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# train data
print("Mean square error for the ARMA(3,0) method forecasting on Electricity (Wh) is\n ", mse(arma_1_t_error).round(4))
print(sm.stats.acorr_ljungbox(arma_1_t_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 414.214595  with a p-value of 2.560013e-87 ')
print('the mean for the ARMA(3,0) model error is\n', np.mean(arma_1_t_error))
print('the variance for the ARMA(3,0) model error is\n', np.var(arma_1_t_error))
print("the covariance Matrix for the data is\n", model_1.cov_params())
print("The standard error coefficients are \n", model_1.bse)
print('The confidence intervals are\n', model_1.conf_int())
print('the RMSE for the ARMA(3,0) model error is\n ', mean_squared_error(df_y_t_n['Appliances'], arma_1_pred_t.values, squared=False))


# forecasting data
print('The MSE for the forecasting data was found to be',mse(arma_1_f_error))
print(sm.stats.acorr_ljungbox(arma_1_f_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 4824.96647  with a p-value of 0.0 ')
print('the mean for the ARMA(3,0) model forecasting error is\n', np.mean(arma_1_f_error))
print('the variance for the ARMA(3,0) model forecasting error is\n', np.var(arma_1_f_error))
print('the RMSE for the ARMA(3,0) model error is\n ', mean_squared_error(df_y_f_n['Appliances'], arma_1_pred_f.values, squared=False))
print('The variance of the prediction error was less than the variance of the forecasting error')

plt.figure()
plt.title('ARMA(3,0) for Electricity (Wh)')
plt.plot(df_y_t_n.index, df_y_t_n.Appliances, color= 'red', label='Train Data')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color= 'green', label='Test Data')
plt.plot(df_y_t_n.index, arma_1_pred_t, color= 'yellow', label='Prediction Data')
plt.plot(df_y_f_n.index, arma_1_pred_f, color= 'blue', label='Forecasting Data')
plt.xticks(df.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.title('Forecasting on ARMA(3,0) for Electricity (Wh)')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color= 'red', label='Test Data')
plt.plot(df_y_f_n.index, arma_1_pred_f, color= 'yellow', label='Forecasting Data')
plt.xticks(df_y_f_n.index[::750], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()




# test for ARMA(3,1)
na = 3
nb = 1
# ARMA model
model_2 = sm.tsa.ARMA(df_y_t_n.Appliances,(na, nb)).fit(trend='nc',disp=0)
print(model_2.summary())


# training calcualtion
arma_2_pred_t = model_2.predict(start=0, end=15787)
arma_2_error = df_y_t_n.Appliances - arma_2_pred_t.values

# forecasting calculation
arma_2_pred_f = model_2.predict(start=15788, end=19734)
arma_2_f_error = df_y_f_n.Appliances - arma_2_pred_f.values


# acf plot of data
# train
plt.figure()
bound = 1.96/np.sqrt(len(arma_2_error))
ACF_Plot(autocorr(arma_2_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARMA (3,1) ACF Plot of Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# test
plt.figure()
bound = 1.96/np.sqrt(len(arma_2_f_error))
ACF_Plot(autocorr(arma_2_f_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARMA (3,1) ACF Plot Forecasting Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# training data
print("Mean square error for the ARMA(3,1) method prediction on Electricity (Wh) is\n ", mse(arma_2_error).round(4))
print(sm.stats.acorr_ljungbox(arma_2_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 24.179986 with a p-value of 0.0002')
print('the mean for the ARMA(3,1) model prediction error is\n', np.mean(arma_2_error))
print('the variance for the ARMA(3,1) model prediction error is\n', np.var(arma_2_error))
print("the covariance Matrix for the data is\n", model_2.cov_params())
print('the standard error coefficients are', model_2.bse)
print('The confidence intervals are\n', model_2.conf_int())
print('the RMSE for the ARMA(3,1) model error is\n ', mean_squared_error(df_y_t_n['Appliances'], arma_2_pred_t.values, squared=False))

# testing data
print('The MSE for the forecasting data was found to be',mse(arma_2_f_error))
print(sm.stats.acorr_ljungbox(arma_2_f_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 4809.263056  with a p-value of 0.0 ')
print('the mean for the ARMA(3,1) model forecasting error is\n', np.mean(arma_2_f_error))
print('the variance for the ARMA(3,1) model forecasting error is\n', np.var(arma_2_f_error))
print('the RMSE for the ARMA(3,1) model error is\n ', mean_squared_error(df_y_f_n['Appliances'], arma_2_pred_f.values, squared=False))
print('The variance of the prediction error was less than the variance of the forecasting error')

plt.figure()
plt.title('ARMA(3,1) for Electricity (Wh)')
plt.plot(df_y_t_n.index, df_y_t_n.Appliances, color= 'red', label='Train Data')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color= 'green', label='Test Data')
plt.plot(df_y_t_n.index, arma_2_pred_t, color= 'yellow', label='Prediction Data')
plt.plot(df_y_f_n.index, arma_2_pred_f, color= 'blue', label='Forecasting Data')
plt.xticks(df.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.title('Forecasting on ARMA(3,1) for Electricity (Wh)')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color= 'red', label='Test Data')
plt.plot(df_y_f_n.index, arma_2_pred_f, color= 'yellow', label='Forecasting Data')
plt.xticks(df_y_f_n.index[::750], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()


# test for ARMA(3,3)
na = 3
nb = 3
# ARMA model
model_3 = sm.tsa.ARMA(df_y_t_n.Appliances,(na, nb)).fit(trend='nc',disp=0)
print(model_3.summary())

# training calcualtion
arma_3_pred_t = model_3.predict(start=0, end=15787)
arma_3_error = df_y_t_n.Appliances - arma_3_pred_t.values

# forecasting calculation
arma_3_pred_f = model_3.predict(start=15788, end=19734)
arma_3_f_error = df_y_f_n.Appliances - arma_3_pred_f.values


#plot of data
plt.figure()
bound = 1.96/np.sqrt(len(arma_3_error))
ACF_Plot(autocorr(arma_3_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARMA(3,3) ACF Forecasting Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# test
plt.figure()
bound = 1.96/np.sqrt(len(arma_3_f_error))
ACF_Plot(autocorr(arma_3_f_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARMA (3,3) ACF Plot Forecasting Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()



# statistics
print("Mean square error for the ARMA(3,3) method forecasting on Electricity (Wh) is\n ", mse(arma_3_error).round(4))
print(sm.stats.acorr_ljungbox(arma_3_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 0.637048 with a p-value of 0.0')
print('the mean for the ARMA(3,3) model error is\n', np.mean(arma_3_error))
print('the variance for the ARMA(3,3) model error is\n', np.var(arma_3_error))
print("the covariance Matrix for the data is\n", model_3.cov_params())
print('The standard error coefficient are', model_3.bse)
print('The confidence intervals are\n', model_3.conf_int())
print('the RMSE for the ARMA(3,3) model error is\n ', mean_squared_error(df_y_t_n['Appliances'], arma_3_pred_t.values, squared=False))

print('The MSE for the forecasting data was found to be',mse(arma_3_f_error))
print(sm.stats.acorr_ljungbox(arma_3_f_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 4820.005256 with a p-value of 0.0 ')
print('the mean for the ARMA(3,3) model forecasting error is\n', np.mean(arma_3_f_error))
print('the variance for the ARMA(3,3) model forecasting error is\n', np.var(arma_3_f_error))
print('the RMSE for the ARMA(3,3) model error is\n ', mean_squared_error(df_y_f_n['Appliances'], arma_3_pred_f.values, squared=False))
print('The variance of the prediction error was less than the variance of the forecasting error')


plt.figure()
plt.title('ARMA(3,3) for Electricity (Wh)')
plt.plot(df_y_t_n.index, df_y_t_n.Appliances, color= 'red', label='Train Data')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color= 'green', label='Test Data')
plt.plot(df_y_t_n.index, arma_3_pred_t, color= 'yellow', label='Prediction Data')
plt.plot(df_y_f_n.index, arma_3_pred_f, color= 'blue', label='Forecasting Data')
plt.xticks(df.index[::4500], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.title('Forecasting on ARMA(3,3) for Electricity (Wh)')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color= 'red', label='Test Data')
plt.plot(df_y_f_n.index, arma_3_pred_f, color= 'yellow', label='Forecasting Data')
plt.xticks(df_y_f_n.index[::750], fontsize= 10)
plt.ylabel('Electricity (Wh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()




###############
# SARIMA model
###############


model_s = sm.tsa.statespace.SARIMAX(df_y_t_n.Appliances, order=(3,0,0), seasonal_order=(0,3,0,12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,)
res = model_s.fit()
print(res.summary())

#train data
ST_pred_t = res.get_prediction(start=pd.to_datetime('2016-01-11 17:00:00'), end=pd.to_datetime('2016-04-30 08:10:00'), dynamic=False)
ST_pred = ST_pred_t.predicted_mean
pred_plot= ST_pred
ST_error = df_y_t_n.Appliances - ST_pred.values

# test data
#ST_pred_f = res.get_prediction(start=pd.to_datetime('2016-04-30 08:20:00'), end=pd.to_datetime('2016-05-27 18:00:00 '), dynamic=False)
# ST_pred_f = res.get_forecast(steps=3947, index=df_y_f_n.index)
ST_pred_f = res.predict(start=0, end=(len(df_y_f_n['Appliances'])))
# ST_pred_f = ST_pred_f.predicted_mean
# pred_f_plot= ST_pred_f
ST_error_f = df_y_f_n.Appliances - ST_pred_f.values[1:]

# train
plt.figure()
bound = 1.96/np.sqrt(len(ST_error))
ACF_Plot(autocorr(ST_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARIMA(3,0,0)x(0,3,0,12) ACF Plot with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# test
plt.figure()
bound = 1.96/np.sqrt(len(ST_error_f))
ACF_Plot(autocorr(ST_error_f,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ARIMA(3,0,0)x(0,3,0,12) ACF Plot Forecasting with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


# training statistics
print("Mean square error for the  ARIMA(3,0,0)x(0,0,0,12) method forecasting on Electricity (Wh) is\n ", mse(ST_error).round(4))
print(sm.stats.acorr_ljungbox(ST_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 78.976724  with a p-value of 1.373662e-15')
print('the mean for the  ARIMA(3,0,0)x(0,3,0,12) model error is\n', np.mean(ST_error))
print('the variance for the  ARIMA(3,0,0)x(0,3,0,12) model error is\n', np.var(ST_error))
print("the covariance Matrix for the data is\n", res.cov_params())
print("the Standard Error for the coefficients are is\n", res.bse)
print('The confidence intervals are\n', res.conf_int())
print('the RMSE for the ARIMA(3,0,0)x(0,3,0,12) model error is\n ', mean_squared_error(df_y_t_n.Appliances, ST_pred.values, squared=False))

# testing statistics
print('The MSE for the forecasting data was found to be',mse(ST_error_f))
print(sm.stats.acorr_ljungbox(ST_error_f, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 923.111417 with a p-value of 2.648581e-197')
print('the mean for the ARIMA(3,0,0)x(0,3,0,12) model forecasting error is\n', np.mean(ST_error_f))
print('the variance for the ARIMA(3,0,0)x(0,3,0,12) model forecasting error is\n', np.var(ST_error_f))
print('the RMSE for the ARIMA(3,0,0)x(0,3,0,12) model error is\n ', mean_squared_error(df_y_f_n['Appliances'], ST_pred_f[1:].values, squared=False))
print('the variance of the prediction error appeared less than the variance of the forecasting error')
# without forecasting
plt.figure()
plt.title('ARIMA(3,0,0)x(0,3,0,12) on Electricity (Wh)')
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.plot(df_y_t_n.index, pred_plot, color ='yellow', label='Prediction Data')
plt.plot(df_y_t_n.index, df_y_t_n.Appliances, color='red', label='Train Data')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color='green', label='Test Data')
# plt.plot(df_y_f_n.index, pred_f_plot, color ='blue', label='Forecasting Data')
plt.xticks(df_y.index[::4500], rotation= 90, fontsize= 10)
plt.legend()
plt.tight_layout()
plt.show()


# with forecasting
plt.figure()
plt.title('ARIMA(3,0,0)x(0,3,0,12) on Electricity (Wh)')
plt.xlabel('Date')
plt.ylabel('Electricity (Wh)')
plt.plot(df_y_t_n.index, pred_plot, color ='yellow', label='Prediction Data')
plt.plot(df_y_t_n.index, df_y_t_n.Appliances, color='red', label='Train Data')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances, color='green', label='Test Data')
plt.plot(df_y_f_n.index, ST_pred_f[1:], color ='blue', label='Forecasting Data')
plt.xticks(df_y.index[::4500], rotation= 90, fontsize= 10)
plt.legend()
plt.tight_layout()
plt.show()


#14: base models
def one_step_average_method(x):
    x_br = []
    for i in range(1,len(x)):
        m = np.mean(x[0:i])
        x_br.append(m)
    return x_br

one_step_average_method(df_y_t_n.Appliances)

def h_step_average_method(train, test):
    forecast = np.mean(train)
    predictions = []
    for i in range(len(test)):
        predictions.append(forecast)
    return predictions
H_avg= h_step_average_method(df_y_t_n['Appliances'],df_y_f_n['Appliances'])


# Q2 plot train, test, and average

plt.figure()
plt.plot(df_y_t_n.index,df_y_t_n.Appliances,label= "Train Data", color = 'blue')
plt.xticks(df.index[::4500], rotation= 90, fontsize= 10)
plt.plot(df_y_f_n.index,df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, H_avg, label = 'Average Method', color = 'yellow')
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Average Method on Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df_y_f_n.index,df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, H_avg, label = 'Average Method', color = 'yellow')
plt.xticks(df_y_f_n.index[::725], fontsize= 10)
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Average Method on Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()



#train
avg_yt_error = np.array(df_y_t_n.Appliances[1:]) - np.array(one_step_average_method(df_y_t_n.Appliances))
print("Mean square error for the average method training set is ", mse(avg_yt_error).round(4))
#forecast
avg_yf_error = np.array(df_y_f_n.Appliances) - np.array(H_avg)
print("Mean square error for the average method testing set is ", mse(avg_yf_error).round(4))

# Average method statistics
print('the variance of the error of the average method training set is ', np.var(avg_yt_error))
print('the variance of the error of the average method testing set is ', np.var(avg_yf_error))
print('the RMSE of the Average forecasting model error is, ', mean_squared_error(df_y_f_n['Appliances'], np.array(H_avg), squared=False))
print('the mean of the Average forecasting  model error is', np.mean(avg_yf_error))
print(sm.stats.acorr_ljungbox(avg_yf_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 5014.178759 with a p-value of 0.0')
print('the varaince of hte prediction error appeared larger than the variance of the forecasting error')

plt.figure()
bound = 1.96/np.sqrt(len(df_y_t_n.Appliances))
ACF_Plot(autocorr(avg_yf_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Average Method Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


# naive method
def one_step_naive_method(x):
    forecast = []
    for i in range(len(x)-1):
        forecast.append(x[i])
    return forecast

def h_step_naive_method(test,train):
    forecast = [test[-1] for i in range (len(train))]
    return forecast
N_avg= h_step_naive_method(df_y_t_n.Appliances,df_y_f_n.Appliances)



plt.figure()
plt.plot(df_y_t_n.index,df_y_t_n.Appliances,label= "Train Data", color = 'blue')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, N_avg, label = 'Naive Method', color = 'yellow')
plt.xticks(df.index[::4500], rotation= 90, fontsize= 10)
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Naive Method on Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df_y_f_n.index, df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, N_avg, label = 'Naive Method', color = 'yellow')
plt.xticks(df_y_f_n.index[::725], fontsize= 10)
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Naive Method on Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()


#train
N_yt_error = np.array(df_y_t_n.Appliances[1:]) - np.array(one_step_naive_method(df_y_t_n.Appliances))
print("Mean square error for the Naive method training set is ", mse(N_yt_error))
#forecast
N_yf_error = np.array(df_y_f_n.Appliances) - np.array(N_avg)
print("Mean square error for the Naive method testing set is ", mse(N_yf_error))

# Q6d: Naive method statistics
print('the Variance of the error of the Naive method training set is ', np.var(N_yt_error))
print('the Variance of the error of the Naive method testing set is ', np.var(N_yf_error))
print('the RMSE of the Naive model forecasting error is, ', mean_squared_error(df_y_f_n['Appliances'], np.array(N_avg), squared=False))
print('the mean of the Naive model forecasting error is', np.mean(N_yf_error))
print(sm.stats.acorr_ljungbox(N_yf_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 5014.178759  with a p-value of 0.0')
print('the variance for the prediction error appeared less than the variance of the forecasting error')

plt.figure()
bound = 1.96/np.sqrt(len(df_y_t_n.Appliances))
ACF_Plot(autocorr(N_yf_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Naive Method Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()

# drift method

def one_step_drift_method(x):
    forecast =[]
    for i in range(1,len(x)-1):
        #slope calculation
        prediction = x[i]+(x[i]-x[0])/i
        forecast.append(prediction)
    #gives you every prediction except first bc first is first value of dataset
    forecast = [x[0]] + forecast
    return forecast

one_step_drift_method(df_y_t_n.Appliances)

def h_step_drift_method(train,test):
    forecast = []
    # use first and last number in train to calcuate slope for h step
    prediction = (train[-1] - train[0]) / (len(train)-1)
    for i in range(1,len(test) + 1):
        forecast.append(train[-1]+ i*prediction)
    return forecast
H_drift= h_step_drift_method(df_y_t_n.Appliances,df_y_f_n.Appliances)

# Q7b: plot train, test, and drift

plt.figure()
plt.plot(df_y_t_n.index,df_y_t_n.Appliances,label= "Train Data", color = 'blue')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, H_drift, label = 'Drift Method', color = 'yellow')
plt.xticks(df.index[::4500], rotation= 90, fontsize= 10)
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Drift Method on Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df_y_f_n.index, df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, H_drift, label = 'Drift Method', color = 'yellow')
plt.xticks(df_y_f_n.index[::725], fontsize= 10)
plt.xlabel('Time')
plt.ylabel('Electricity (Wh)')
plt.title('Drift Method on Electricity (Wh)')
plt.legend()
plt.tight_layout()
plt.show()



#train
drift_yt_error = np.array(df_y_t_n.Appliances[1:]) - np.array(one_step_drift_method(df_y_t_n.Appliances))
print("Mean square error for the drift method training set is ", mse(drift_yt_error))
#forecast
drift_yf_error = np.array(df_y_f_n.Appliances) - np.array(H_drift)
print("Mean square error for the drift method testing set is ", mse(drift_yf_error))

# Q7d: drift method variance
# Q6d: Naive method statistics
print('the Variance of the error of the Drift method training set is ', np.var(drift_yt_error))
print('the Variance of the error of the Drift method testing set is ', np.var(drift_yf_error))
print('the RMSE of the Drift model forecasting error is, ', mean_squared_error(df_y_f_n['Appliances'], np.array(H_drift), squared=False))
print('the mean of the Drift model forecasting error is', np.mean(drift_yf_error))
print(sm.stats.acorr_ljungbox(drift_yf_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 5129.25267 with a p-value of 0.0')
print('the variance for the prediction error appeared less than the variance of the forecasting error')

plt.figure()
bound = 1.96/np.sqrt(len(df_y_t_n.Appliances))
ACF_Plot(autocorr(drift_yf_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Drift Method Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()


# Seasonal exponential smoothing

# Q8a: see report

holtt = ets.ExponentialSmoothing(df_y_t_n.Appliances,trend=None,damped_trend=False,seasonal=None).fit(smoothing_level=0.5)
holtf = holtt.forecast(steps=len(df_y_f_n))
holtf = pd.DataFrame(holtf)


#plot train, test, and SES

plt.figure()
plt.plot(df_y_t_n.index,df_y_t_n.Appliances,label= "Train Data", color = 'blue')
plt.plot(df_y_f_n.index, df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, np.array(holtf), label = 'SES Method', color = 'yellow')
plt.xticks(df.index[::4500], rotation= 90, fontsize= 10)
plt.xlabel('Date')
plt.ylabel('Appliances (Wh)')
plt.title('SES Method on Appliances(Wh)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df_y_f_n.index, df_y_f_n.Appliances,label= "Test Data", color = 'red')
plt.plot(df_y_f_n.index, np.array(holtf), label = 'SES Method', color = 'yellow')
plt.xticks(df_y_f_n.index[::725], fontsize= 10)
plt.xlabel('Date')
plt.ylabel('Appliances (Wh)')
plt.title('SES Method on Appliances(Wh)')
plt.legend()
plt.tight_layout()
plt.show()


#train
def SES_train(yt,alpha, initial=430):
    prediction = [initial]
    for i in range(1,len(yt)):
        s= alpha*yt[i-1] + (1-alpha)*prediction[i-1]
        prediction.append(s)
    return prediction

SES_yt_error = np.array(df_y_t_n.Appliances) - SES_train(df_y_t_n.Appliances,.5)
print("Mean square error for the SES method training set is ", mse(SES_yt_error))
#forecast
holtf = holtt.forecast(steps=len(df_y_f_n.Appliances))
holtf = pd.DataFrame(holtf)
SES_yf_error = np.array(df_y_f_n.Appliances) - holtf[0]
print("the mean square error for the SES method testing set is ", mse(SES_yf_error))
print('the Variance of the error of the SES method training set is ', np.var(SES_yt_error))
print('the Variance of the error of the SES method testing set is ', np.var(SES_yf_error))
print('the RMSE of the SES model forecasting error is, ', mean_squared_error(df_y_f_n['Appliances'], holtf[0], squared=False))
print('the mean of the SES model forecasting error is', np.mean(SES_yf_error))
print(sm.stats.acorr_ljungbox(SES_yf_error, lags=[5], boxpierce=True, return_df=True))
print('The Q value was found to be 5014.178759 with a p-value of 0.0')
print('The variance of the prediction error appeared less than the variance of the forecasting error')

plt.figure()
bound = 1.96/np.sqrt(len(df_y_t_n.Appliances))
ACF_Plot(autocorr(SES_yf_error,90),90)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of SES Method Electricity (Wh) with 90 Lags')
plt.axhspan(-bound,bound,alpha = .1, color = 'black')
plt.tight_layout()
plt.show()