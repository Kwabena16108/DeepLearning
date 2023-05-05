#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:17:41 2023

@author: dixondomfeh
"""

import time
import numpy as np
import pandas as pd
import datetime as datetime

import matplotlib.pyplot as plt

import yfinance as yf # Must be v0.1.83: https://github.com/ranaroussi/yfinance/issues/1484
import talib as ta
import pyfolio as pf
import warnings


#
def diff(x, d):
    '''-- this function is for fractional differencing--
    recent innovation from Marco De Lopez
    '''
    x = np.array(x)
    if len(x.shape) > 1 and x.shape[1] > 1:
        raise ValueError("only implemented for univariate time series")
    if np.any(np.isnan(x)):
        raise ValueError("NAs in x")
    n = len(x)
    if n < 2:
        raise ValueError("n must be >= 2")
    x = x - np.mean(x)
    PI = np.zeros(n)
    PI[0] = -d
    for k in range(1, n):
        PI[k] = PI[k - 1] * (k - 1 - d) / k
    ydiff = np.copy(x)
    for i in range(1, n):
        ydiff[i] = x[i] + np.sum(PI[:i] * x[i-1::-1])
    return ydiff


#%% Load minute tick data from yahoo finance

sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
sp500_tickers.sort()


apple_minute_data = yf.download(tickers="AAPL", period="7d", interval="1m", auto_adjust=True)


start="1982-04-09"
end="2023-04-09"


aapl_data = yf.download(tickers="AAPL",interval='1d', start=start, end=end , auto_adjust=True)











































