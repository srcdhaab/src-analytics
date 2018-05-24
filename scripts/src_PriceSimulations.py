# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:38:37 2018

@author: D
"""

#### in this script we create a simulation of the SRC price, approached with classical analysis - pseudo Brownian Motion
## leverage from this site: https://jtsulliv.github.io/stock-movement/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import quandl
import logging
import time
import sys
import csv
import os
import random
################### Params
quandl.ApiConfig.api_key = 'rdYo66CkyYAwWBPLZxjX'
seed=5
years=5
tradingdays=365
quandlDataPath= 'C:/Users/D/Dropbox/Work/SRC/Data/Quandl_temp'
btcusd_exchanges = ['COINBASE','BITSTAMP','ITBIT','KRAKEN']
altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']


################# Quandl Data Function
def get_quandl_data(quandl_id,folderpath):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','_')
    cache_path = '{}/{}'.format(folderpath,cache_path)
    print(cache_path)
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df

def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''

    try:        
        f = open(cache_path, 'rb')
       
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df

def get_crypto_data(poloniex_pair,datafolder):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, '{}/{}'.format(datafolder,poloniex_pair))
    data_df = data_df.set_index('date')
    return data_df

def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)

def BrownianMotion(seed,N):
    #set the seed
    np.random.seed(seed)
    #set the timesteps per unit time
    dt=1./N 
    # W_t-w_t-dt = sqrt(dt)*Z_t where Z~N(0,1)
    brownianIncrement = np.random.normal(0., 1., int(N))*np.sqrt(dt) 
    # The brownian motion's trajectory is the sum of all the increments
    W = np.cumsum(brownianIncrement) 
    #A BM starts at zero W_0=0
    #W = np.insert(W,0,0.)
    
    return brownianIncrement,W

### solution to the stock diff equation dS=S*r*dt + sigma*dW is S_t=S_0*exp((r-sigma^2/2)t+sigma*W_t)
## 
def AssetPriceByBMOverT(mu,sigma,N,S0,seed):
    T=1.
    t=np.linspace(0.,T,N)
    W = BrownianMotion(seed,N)[1]
    S_t=[S0]
    
    for i in np.arange(0,N-1,1):
        S_t.append(S0*np.exp((mu-0.5*sigma**2)*t[i]+sigma*W[i]))
        
    return S_t
    
def InnerValueMovementPerToken1Factor(mu,N,V0,seed,yearlyRevenue,list_Gatetime=[],list_COValue=[]):
    #This functions models the evolution of the inner value
    #mu:        returns from RE
    #varcovar:  varcovar matrix of the returns of (RE,IP) 
    #N:         Number steps in the unit-time
    #seed:      Seed for the randomness
    T=1.
    V_t=[V0]
    r_ip = IntellectualPropertyRevenueStreamPerToken(yearlyRevenue,N)
    #runner for the lists
    if len(list_Gatetime)==0:
        list_Gatetime=[0]
        list_COValue=[0]
    j=0
    for i in np.arange(1,N,1):
        
        if i == list_Gatetime[min(j,len(list_Gatetime)-1)]:
            #adding value through the gating mechanism
            V_t.append(V_t[i-1]*(1+mu/365)**(1./365)+list_COValue[j]+r_ip[i])
            j=j+1
        else:
            V_t.append(V_t[i-1]*(1+mu/365)**(1./365)+r_ip[i])
        
        
    return V_t
        
    
def IntellectualPropertyRevenueStreamPerToken(yearlyRevenue,N):
    #simulates a trajectory of IP Revenue streams
    #yearlyRevenue:  N/365x1 Vector of projected IP revenue per token that wll be linearized to a daily stream
    days = 365 #not trading days but actual days
    yearlyRevenue=(np.asarray(yearlyRevenue))/days
    IP=[]
    assert (N/365<=len(yearlyRevenue)),"Provide a longer yearly return time horizon"
    for i in np.arange(0,len(yearlyRevenue[0:int(N/365)]),1):
        
        IP.append((yearlyRevenue[i]*np.ones(days)+np.random.normal(-0.02/days,yearlyRevenue[i]/days,days)).flatten())
    
        
    
    return np.concatenate(IP).ravel().tolist()
####comment out - for testing
V_t =  InnerValueMovementPerToken1Factor(mu,simlength,V0,seed,yearlyRevenue)
df_p = pd.DataFrame.from_dict({"V_t":V_t,"S_t":temp_trajectory,"N_t": 100.*np.ones_like(V_t)})   
 ###   
   
def createTrajectoryWithSRCMechanism(x_days,df_p,mu,V0,seed,list_COValue=[],list_Gatetime=[]):
    ''' crude implementation of the price mechanism '''
    # xdays:  integer that defines the time window of the mavg
    # df_p:   dataframe with columns S_t (Tokenprice), V_t (inner value), N_t (Tokensupply)
    
    #x-day moving average column
    df_p['{}d-mavg'.format(x_days)] = df_p['S_t'].rolling(x_days,1).mean()
    #Implement gating mechanism
    df_p['Criteria'] = np.greater(df_p['{}d-mavg'.format(x_days)],2*df_p['V_t'])
    #Find the first  time the criteria is met
    N=len(df_p)
    if df_p['Criteria'][df_p['Criteria']==True].empty:
        # if the criteria is never met set the crit date to the end
        t_crit = df_p.shape[0]
        
    else:
        # set the time to the index of the first occurence of true
        t_crit = df_p['Criteria'][df_p['Criteria']==True].index[0]
        # Select the next 3 days to sell new tokens at market price i.e. p(crit=true)
        p = df_p['S_t'].iloc[t_crit]
        # first step assume a doubling ( complexify later)
        sold_fraction=1.
        # add it to the inner value as t+3 = (S[t]+V[t])/2-V_t = (S[t]-V[t])/2
        additionalInnerVal = (p - df_p['V_t'].iloc[t_crit])/2.
        #add it to the list of added values
        list_COValue.append(additionalInnerVal)
        list_Gatetime.append(t_crit)
        print("p={},addVal={},t_crit={}".format(p,additionalInnerVal,t_crit))
#        V_t = InnerValueMovementPerToken1Factor(mu,N,V0,seed,yearlyRevenue,t_crit,additionalInnerVal)
        #overwrite the former inner value column with the new column
#        print("l(df)={}, l(v_t)={})".format(len(df_p),len(V_t)))
        df_p['V_t'] = InnerValueMovementPerToken1Factor(mu,N,V0,seed,yearlyRevenue,list_Gatetime,list_COValue)
        #update the token supply
        df_p['N_t'].iloc[t_crit:]= df_p['N_t'].iloc[t_crit:] + sold_fraction * df_p['N_t'].iloc[t_crit:]
        ##################
    # and recursively call the function
    
        if t_crit < (df_p.shape[0]-1):
            print("Gate Mechanism opens at t={}, total length of sample is T={}".format(t_crit,df_p.shape[0]-1))
            df_p = createTrajectoryWithSRCMechanism(x_days,df_p,mu,V0,seed)
        
                    
    #return the dataframe with price and value asf
    return df_p
    
#########################################
## Get the daily BTC quotes using quandl
exchange_data = {}
classical_data = {}

for exchange in btcusd_exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    print(exchange_code)
    btc_exchange_df = get_quandl_data(exchange_code,quandlDataPath)
    exchange_data[exchange] = btc_exchange_df



btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')
### Remove "0" values
btc_usd_datasets.replace(0, np.nan, inplace=True)

### Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)
df_length=len(btc_usd_datasets)
btc_usd_datasets.insert(0,'Row_ID',range(0,0+df_length))    


#exchange_data
#define the average daily return as 'return'
btc_usd_datasets['DailyReturn']=btc_usd_datasets['avg_btc_price_usd'].pct_change()+1
btc_usd_datasets.head()


##################################
seed=5
N = 128
##### Create an example path of the inner value and the price
b = BrownianMotion(seed,N)[0]
W = BrownianMotion(seed,N)[1]
#
#
## brownian increments
##%matplotlib inline
#plt.rcParams['figure.figsize'] = (10,8)
#plt.figure()
#xb = np.linspace(1, len(b), len(b))
#plt.plot(xb, b)
#plt.title('Brownian Increments')
#
## brownian motion
#xw = np.linspace(1, len(W), len(W))
#plt.figure()
#plt.plot(xw, W)
#plt.title('Brownian Motion')
#  

#set the seed
seed = 2018
#
N=2*365
seed=5
S0=1.
mu=0.2
sigma=0.6

mu_re=0.04
V0 = 0.8
yearlyRevenue = [0.03,0.07,0.1,0.2,0.37,0.5]
S_t = AssetPriceByBMOverT(mu,sigma,N,S0,seed)
V_t = InnerValueMovementPerToken1Factor(mu_re,N,V0,seed,yearlyRevenue)



plt.figure()
t=np.linspace(0,len(S_t),len(S_t))
plt.plot(t,S_t)
plt.figure()
plt.plot(np.linspace(0,len(V_t),len(V_t)),V_t)








###########
## retrieve the return and volatility for RE investments
REdata = r'C:\Users\D\Dropbox\Work\SRC\Data\SIX_RealEstateBroad.xlsx'
RE_df =  pd.read_excel(REdata, sheet_name='SIX_clean')
RE_df=RE_df.set_index(pd.to_datetime(RE_df['Date'], format='%d.%m.%y'))
RE_df=RE_df.drop(['Date'], axis=1)
RE_df=RE_df.rename(columns = lambda x : str(x)[len(r'SXI Real Estate '):])

RE_df=RE_df.pct_change()
RE_df.head()
averages=RE_df.mean()

##
########
## 
price_df = pd.DataFrame.from_dict({'S_t':S_t,'V_t':V_t})
price_df.head()

#######################################
## draw random returns out of the realized BTC returns
df_2=btc_usd_datasets[['Row_ID','DailyReturn']]
MCSim=[]
df_2=btc_usd_datasets[['Row_ID','BTCDailyReturn']]
df_2=df_2.set_index('Row_ID')
src_mcSimyears=3
simlength=365*src_mcSimyears
V_t =  InnerValueMovementPerToken1Factor(mu,simlength,V0,seed,yearlyRevenue)
x_days=30
#For the number of Simulations draw a random set of return from the realized BTC returns with replacement
for j in np.arange(0,3,1):#Number_MCSims,1):
    #draw the returns
    temp_returns=df_2.iloc[random.sample(range(1,df_length),simlength),0].values
    #save the returns
    MCSim.append(temp_returns)
    #determine the trajectory
    temp_trajectory = np.cumprod(temp_returns)
    ###### simulate our mechanism
    #use a xday mavg of the price and compare it to the inner value
    #Save the Value and SRC series into a DF
    
    df_p = pd.DataFrame.from_dict({"V_t":V_t,"S_t":temp_trajectory,"N_t": 100.*np.ones_like(V_t)})
    
    df_p = createTrajectoryWithSRCMechanism(x_days,df_p,mu,V0,seed)
    
    plt.plot( df_p.index, 'S_t', data=df_p,color='skyblue')
    plt.plot( df_p.index, 'V_t', data=df_p,color='red')

number_Bins=50       
plt.figure()
np.histogram(df_2['DailyReturn'].iloc[1:],bins=number_Bins)    
 
plt.hist(df_2['DailyReturn'].iloc[1:]-1,bins=number_Bins)

#    print('Length of traj is {}'.format(len(temp_trajectory)))
#    #plot the trajectory
#    xb = np.linspace(0, len(temp_trajectory)-1, len(temp_trajectory))
#    plt.figure()
#    plt.plot(xb, temp_trajectory)
    

