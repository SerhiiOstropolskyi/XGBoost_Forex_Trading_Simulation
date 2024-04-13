# Generalization with OOP: The Tools Class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from itertools import product
plt.style.use("seaborn-v0_8")

# import datetime as dt
# from dateutil.relativedelta import relativedelta
# import yfinance as yf

# Moving_Averages
class Moving_Averages():
    def __init__(self, type, period, data, EMA_min_periods = 0):
        self.type = type
        self.period = period
        self.data = data
        self.EMA_min_periods = EMA_min_periods
        self.result = pd.DataFrame()
        self.calculate_data()
        
    def __repr__(self):
        return "Moving_Averages(type = {}, period = {}, data, EMA_min_periods = {})".format(self.type, self.period, self.EMA_min_periods)
    
    def calculate_data(self):
        #data = self.data.copy()
        data = pd.DataFrame()
        if self.type == "SMA":
            data["SMA-{}".format(self.period)] = self.calculate_SMA(self.data, self.period)
        elif self.type == "EMA":
            data["EMA-{}".format(self.period)] = self.calculate_EMA(self.data, self.period)
        elif self.type == "WMA":
            data["WMA-{}".format(self.period)] = self.calculate_WMA(self.data, self.period)
        elif self.type == "HMA":
            data["HMA-{}".format(self.period)] = self.calculate_HMA(self.data, self.period)
        elif self.type == "MA":
            data["SMA-{}".format(self.period)] = self.calculate_SMA(self.data, self.period)
            data["EMA-{}".format(self.period)] = self.calculate_EMA(self.data, self.period)
            data["WMA-{}".format(self.period)] = self.calculate_WMA(self.data, self.period)
            data["HMA-{}".format(self.period)] = self.calculate_HMA(self.data, self.period)
        #data.dropna(inplace = True)
        data = data.dropna()
        self.result = data
        #return data

    def calculate_SMA(self, data_SMA, period_SMA):
        data = data_SMA.copy()
        data["SMA"] = data.rolling(period_SMA).mean()
        #data.dropna(inplace = True)
        data = data.dropna()
        return data["SMA"]
    
    def calculate_EMA(self, data_EMA, period_EMA):
        data = data_EMA.copy()
        # ewm --> parameter - alpha: float, optional Specify smoothing factor directly, 0 < a <= 1
        data["EMA"] = data.ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean()
        #data_EMA.dropna(inplace = True)
        data_EMA = data_EMA.dropna()
        return data["EMA"]

    def calculate_WMA(self, data_WMA, period_WMA):
        data = data_WMA.copy()
        data["WMA"] = data.rolling(period_WMA).apply(lambda x: ((np.arange(period_WMA)+1)*x).sum()/(np.arange(period_WMA)+1).sum(), raw=True)
        #data_WMA.dropna(inplace = True)
        data_WMA = data_WMA.dropna()
        return data["WMA"]

    def calculate_HMA(self, data_HMA, period_HMA):        
        data = data_HMA.copy()
        a = self.calculate_WMA(data.copy(), period_HMA//2).multiply(2)
        b = self.calculate_WMA(data.copy(), period_HMA)
        c = a - b
        data["HMA"] = self.calculate_WMA(c, int(np.sqrt(period_HMA)))
        #data_HMA.dropna(inplace = True)
        data_HMA = data_HMA.dropna()
        return data["HMA"]

    def plot_MA(self):
        title = "type = {} | period = {}".format(self.type, self.period)
        #self.data["{}".format(self.type)].plot(title=title, figsize=(12, 8))
        

# Stochastict
class Stochastic():
    def __init__(self, K_period, K_slow_period, D_period, D_slow_period, data,
                 K_slow_smoothing_type = None, D_smoothing_type = None, D_slow_smoothing_type  = None, EMA_min_periods = 0):
        self.K = None
        self.K_slow = None
        self.D0 = None
        self.D1 = None
        self.D_slow = None
        self.K_period = K_period
        self.K_slow_period = K_slow_period
        self.D_period = D_period
        self.D_slow_period = D_slow_period
        self.K_slow_smoothing_type = K_slow_smoothing_type
        self.D_smoothing_type = D_smoothing_type
        self.D_slow_smoothing_type = D_slow_smoothing_type
        self.data = data
        self.EMA_min_periods = EMA_min_periods
        self.results = pd.DataFrame()
        self.calculate_data()
        
    def __repr__(self):
        return "Stochastic(K_period = {}, K_slow_period = {}, D_period = {}, D_slow_period = {},  K_slow_smoothing_type = {}, D_smoothing_type = {}, D_slow_smoothing_type = {}, data)".format( self.K_period, self.K_slow_period, self.D_period, self.D_slow_period, self.K_slow_smoothing_type, self.D_smoothing_type, self.D_slow_smoothing_type)
    # Averages
    def calculate_WMA(self, data_WMA, period_WMA):
        data = data_WMA.copy()
        data["WMA"] = data.rolling(period_WMA).apply(lambda x: ((np.arange(period_WMA)+1)*x).sum()/(np.arange(period_WMA)+1).sum(), raw=True)
        return data["WMA"]

    def calculate_HMA(self, data_HMA, period_HMA):        
        data = data_HMA.copy()
        a = self.calculate_WMA(data.copy(), period_HMA//2).multiply(2)
        b = self.calculate_WMA(data.copy(), period_HMA)
        c = a - b
        data["HMA"] = self.calculate_WMA(c, int(np.sqrt(period_HMA)))
        return data["HMA"]
    
    # Stochastic calculation
    def calculate_data(self):
        #SO K line:
        min = self.data.Low.rolling(self.K_period).min()
        max = self.data.High.rolling(self.K_period).max()
        self.K = ((self.data.Close - min) / (max - min))*100
        self.results[f"%K-{self.K_period}"] = self.K
        
        #SO K slow line:
        if self.K_slow_smoothing_type == "SMA":
            # av_min = self.data.Low.rolling(self.K_slow_period).mean()
            # av_max = self.data.High.rolling(self.K_slow_period).mean()
            # av_close = self.data.Close.rolling(self.K_slow_period).mean()
            av_min = min.rolling(self.K_slow_period).mean()
            av_max = max.rolling(self.K_slow_period).mean()
            av_close = self.data.Close.rolling(self.K_slow_period).mean()

        elif self.K_slow_smoothing_type == "EMA":
            # av_min = self.data.Low.ewm(span = self.K_slow_period, adjust=False).mean()
            # av_max = self.data.High.ewm(span = self.K_slow_period, adjust=False).mean()
            # av_close = self.data.Close.ewm(span = self.K_slow_period, adjust=False).mean()
            av_min = min.ewm(span = self.K_slow_period, adjust=False, min_periods = self.EMA_min_periods).mean()
            av_max = max.ewm(span = self.K_slow_period, adjust=False, min_periods = self.EMA_min_periods).mean()
            av_close = self.data.Close.ewm(span = self.K_slow_period, adjust=False, min_periods = self.EMA_min_periods).mean()

        elif self.K_slow_smoothing_type == "WMA":
            # av_min = self.calculate_WMA(self.data.Low, self.K_slow_period)
            # av_max = self.calculate_WMA(self.data.High, self.K_slow_period)
            # av_close = self.calculate_WMA(self.data.Close, self.K_slow_period)
            av_min = self.calculate_WMA(min, self.K_slow_period)
            av_max = self.calculate_WMA(max, self.K_slow_period)
            av_close = self.calculate_WMA(self.data.Close, self.K_slow_period)

        elif self.K_slow_smoothing_type == "HMA":
            av_min = self.calculate_HMA(self.data.Low, self.K_slow_period)
            av_max = self.calculate_HMA(self.data.High, self.K_slow_period)
            av_close = self.calculate_HMA(self.data.Close, self.K_slow_period)
        
        self.K_slow = ((av_close - av_min) / (av_max - av_min))*100
        self.results["{}-%K-{}".format(self.K_slow_smoothing_type, self.K_slow_period)] = self.K_slow
        
        #SO D0 line from K:
        if self.D_smoothing_type == "SMA":
            self.D0 = self.K.rolling(self.D_period).mean()
            self.results[f"SMA-%D0-{self.D_period}"] = self.D0
            
        elif self.D_smoothing_type == "EMA":
            self.D0 = self.K.ewm(span = self.D_period, adjust=False, min_periods = self.EMA_min_periods).mean()
            self.results[f"EMA-%D0-{self.D_period}"] = self.D0

        elif self.D_smoothing_type == "WMA":
            self.D0 = self.calculate_WMA(self.K, self.D_period)
            self.results[f"WMA-%D0-{self.D_period}"] = self.D0
            
        elif self.D_smoothing_type == "HMA":
            self.D0 = self.calculate_HMA(self.K, self.D_period)
            self.results[f"HMA-%D0-{self.D_period}"] = self.D0

        #SO D1 line from K slow: 
        if self.D_smoothing_type == "SMA" and self.K_slow_smoothing_type is not None:
            self.D1 = self.K_slow.rolling(self.D_period).mean()
            self.results[f"SMA-%D1-{self.D_period}"] = self.D1
            
        elif self.D_smoothing_type == "EMA" and self.K_slow_smoothing_type is not None:
            self.D1 = self.K_slow.ewm(span = self.D_period, adjust=False, min_periods = self.EMA_min_periods).mean()
            self.results[f"EMA-%D1-{self.D_period}"] = self.D1

        elif self.D_smoothing_type == "WMA" and self.K_slow_smoothing_type is not None:
            self.D1 = self.calculate_WMA(self.K_slow, self.D_period)
            self.results[f"WMA-%D1-{self.D_period}"] = self.D1
            
        elif self.D_smoothing_type == "HMA" and self.K_slow_smoothing_type is not None:
            self.D1 = self.calculate_HMA(self.K_slow, self.D_period)
            self.results[f"HMA-%D1-{self.D_period}"] = self.D1
        
        #SO D slow line:
        if self.D_slow_smoothing_type == "SMA":
            self.D_slow = self.D0.rolling(self.D_slow_period).mean()
            self.results[f"SMA-%D_slow-{self.D_slow_period}"] = self.D_slow
            
        elif self.D_slow_smoothing_type == "EMA":
            self.D_slow = self.D0.ewm(span = self.D_slow_period, adjust=False,  min_periods = self.EMA_min_periods).mean()
            self.results[f"EMA-%D_slow-{self.D_slow_period}"] = self.D_slow

        elif self.D_slow_smoothing_type == "WMA":
            self.D_slow = self.calculate_WMA(self.D0, self.D_slow_period)
            self.results[f"WMA-%D_slow-{self.D_slow_period}"] = self.D_slow
            
        elif self.D_slow_smoothing_type == "HMA":
            self.D_slow = self.calculate_HMA(self.D0, self.D_slow_period)
            self.results[f"HMA-%D_slow-{self.D_slow_period}"] = self.D_slow
        
        #self.results.dropna(inplace = True)
        self.results = self.results.dropna()

    # Plot
    def plot_SO(self):
        self.results.plot(figsize=(12, 8))

# RSI
class RSI():
    def __init__(self, type, period, data, EMA_min_periods = 0):
        self.type = type
        self.period = period
        self.data = data
        self.prepared_data = pd.DataFrame()
        self.results = pd.DataFrame() 
        self.EMA_min_periods = EMA_min_periods
        self.prepare_data()
        self.calculate_data()
        
    def __repr__(self):
        return "RSI(type = {}, period = {}, data)".format(self.type, self.period)
    
    def calculate_data(self):
        data = self.data.copy()
        if self.type == "SMA":
            data[f"RSI-SMA-{self.period}"] = self.calculate_RSI_SMA(self.prepared_data, self.period)
            self.results[f"RSI-SMA-{self.period}"] = data[f"RSI-SMA-{self.period}"]
            #self.results.dropna(inplace = True)
            self.results = self.results.dropna()
        elif self.type == "EMA":
            data[f"RSI-EMA-{self.period}"] = self.calculate_RSI_EMA(self.prepared_data, self.period)
            self.results[f"RSI-EMA-{self.period}"] = data[f"RSI-EMA-{self.period}"]
            #self.results.dropna(inplace = True)
            self.results = self.results.dropna()
        elif self.type == "WMA":
            data[f"RSI-WMA-{self.period}"] = self.calculate_RSI_WMA(self.prepared_data, self.period)
            self.results[f"RSI-WMA-{self.period}"] = data[f"RSI-WMA-{self.period}"]
            #self.results.dropna(inplace = True)
            self.results = self.results.dropna()
        elif self.type == "HMA":
            data[f"RSI-HMA-{self.period}"] = self.calculate_RSI_HMA(self.prepared_data, self.period)
            self.results[f"RSI-HMA-{self.period}"] = data[f"RSI-HMA-{self.period}"]
            #self.results.dropna(inplace = True)
            self.results = self.results.dropna()
        elif self.type == "MA":
            data[f"RSI-SMA-{self.period}"] = self.calculate_RSI_SMA(self.prepared_data, self.period)
            data[f"RSI-EMA-{self.period}"] = self.calculate_RSI_EMA(self.prepared_data, self.period)
            data[f"RSI-WMA-{self.period}"] = self.calculate_RSI_WMA(self.prepared_data, self.period)
            data[f"RSI-HMA-{self.period}"] = self.calculate_RSI_HMA(self.prepared_data, self.period)
            #self.results = data.drop(["Close"], inplace = True).dropna(inplace = True)
            self.results = data.copy()
            #self.results.drop(columns = ["Close"], inplace = True)
            self.results = self.results.drop(columns = ["Close"])
            #self.results.dropna(inplace = True)
            self.results = self.results.dropna()
        # self.data = data
        # return self.data.dropna(inplace = True)
    
    def prepare_data(self):
        self.prepared_data = self.data.copy()
        # Calculate Price Differences
        self.prepared_data['diff'] = self.prepared_data.diff(1)
        # Calculate Avg. Gains/Losses
        self.prepared_data['gain'] = self.prepared_data['diff'].clip(lower=0)
        self.prepared_data['loss'] = self.prepared_data['diff'].clip(upper=0).abs()
        #self.prepared_data.dropna(inplace = True)
        self.prepared_data = self.prepared_data.dropna()

    # RSI calculation
    def calculate_RSI_SMA(self, data_SMA, period_SMA):
        data = data_SMA.copy()
        sum = (data["gain"].rolling(period_SMA).mean() + data["loss"].rolling(period_SMA).mean())
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0, data["gain"].rolling(period_SMA).mean(), sum)
        data[f"RSI-SMA-{self.period}"] = 100 * data["gain"].rolling(period_SMA).mean() / avg_sum["avg_sum"]
        return data[f"RSI-SMA-{self.period}"]
    
    def calculate_RSI_EMA(self, data_EMA, period_EMA):
        data = data_EMA.copy()
        sum = (data["gain"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean() + data["loss"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean())
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0,  data["gain"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean(), sum)
        data[f"RSI-EMA-{self.period}"] = 100 * data["gain"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean() /  avg_sum["avg_sum"]
        return data[f"RSI-EMA-{self.period}"]
    
    def calculate_RSI_WMA(self, data_WMA, period_WMA):
        data = data_WMA.copy()
        sum = (self.calculate_WMA(data["gain"], period_WMA) + self.calculate_WMA(data["loss"], period_WMA))
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0, self.calculate_WMA(data["gain"], period_WMA), sum)
        data[f"RSI-WMA-{self.period}"] = 100 * self.calculate_WMA(data["gain"], period_WMA) / avg_sum["avg_sum"]
        return data[f"RSI-WMA-{self.period}"]
    
    def calculate_RSI_HMA(self, data_HMA, period_HMA):
        data = data_HMA.copy()
        sum = (self.calculate_HMA(data["gain"], period_HMA) + self.calculate_HMA(data["loss"], period_HMA))
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0, self.calculate_HMA(data["gain"], period_HMA), sum)
        data[f"RSI-HMA-{self.period}"] = 100 * self.calculate_HMA(data["gain"], period_HMA) / avg_sum["avg_sum"]     
        return data[f"RSI-HMA-{self.period}"]
    
    #Averages
    def calculate_WMA(self, data_WMA, period_WMA):
        data = data_WMA.copy()
        data["WMA"] = data.rolling(period_WMA).apply(lambda x: ((np.arange(period_WMA)+1)*x).sum()/(np.arange(period_WMA)+1).sum(), raw=True)
        return data["WMA"]

    def calculate_HMA(self, data_HMA, period_HMA):        
        data = data_HMA.copy()
        a = self.calculate_WMA(data.copy(), period_HMA//2).multiply(2)
        b = self.calculate_WMA(data.copy(), period_HMA)
        c = a - b
        data["HMA"] = self.calculate_WMA(c, int(np.sqrt(period_HMA)))
        return data["HMA"]
    
    # EMA version of HMA
    # def calculate_HMA(self, data_HMA, period_HMA):        
    #     data = data_HMA.copy()
    #     a = data.ewm(span = period_HMA//2, adjust=False, min_periods = self.EMA_min_periods).mean().multiply(2)
    #     b = data.ewm(span = period_HMA, adjust=False, min_periods = self.EMA_min_periods).mean()
    #     c = a - b
    #     data["HMA"] = c.ewm(span = int(np.sqrt(period_HMA)), adjust=False, min_periods = self.EMA_min_periods).mean()
    #     return data["HMA"]
    
    def plot_RSI(self):
        self.results.plot(figsize=(12, 8))

# Stochastic_RSI
class Stochastic_RSI():
    def __init__(self, RSI_type, RSI_period, K_period, D_period, D_slow_period, D_smoothing_type, data, EMA_min_periods = 0):
        self.RSI_type = RSI_type
        self.RSI_period = RSI_period
        self.K_period = K_period
        self.D_period = D_period
        self.D_slow_period = D_slow_period
        self.D_smoothing_type = D_smoothing_type
        self.data = data
        self.prepared_data = pd.DataFrame()
        self.results_RSI = pd.DataFrame()
        self.results_SO_RSI = pd.DataFrame()  
        self.EMA_min_periods = EMA_min_periods
        self.prepare_data()
        self.get_RSI_data()
        self.calculate_stochastic()
        
    def __repr__(self):
        return "Stochastic_RSI(type = {}, period = {}, data)".format(self.type, self.period)
    
    def prepare_data(self):
        self.prepared_data = self.data.copy()
        # Calculate Price Differences
        self.prepared_data['diff'] = self.prepared_data.diff(1)
        # Calculate Avg. Gains/Losses
        self.prepared_data['gain'] = self.prepared_data['diff'].clip(lower=0)
        self.prepared_data['loss'] = self.prepared_data['diff'].clip(upper=0).abs()
        #self.prepared_data.dropna(inplace = True)
        self.prepared_data = self.prepared_data.dropna()

    # RSI calculation
    def calculate_RSI_SMA(self, data_SMA, period_SMA):
        data = data_SMA.copy()
        sum = (data["gain"].rolling(period_SMA).mean() + data["loss"].rolling(period_SMA).mean())
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0, data["gain"].rolling(period_SMA).mean(), sum)
        data[f"RSI-SMA-{period_SMA}"] = 100 * data["gain"].rolling(period_SMA).mean() / avg_sum["avg_sum"]
        return data[f"RSI-SMA-{period_SMA}"]
    
    def calculate_RSI_EMA(self, data_EMA, period_EMA):
        data = data_EMA.copy()
        sum = (data["gain"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean() + data["loss"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean())
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0,  data["gain"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean(), sum)
        data[f"RSI-EMA-{period_EMA}"] = 100 * data["gain"].ewm(span = period_EMA, adjust=False, min_periods = self.EMA_min_periods).mean() /  avg_sum["avg_sum"]
        return data[f"RSI-EMA-{period_EMA}"]
    
    def calculate_RSI_WMA(self, data_WMA, period_WMA):
        data = data_WMA.copy()
        sum = (self.calculate_WMA(data["gain"], period_WMA) + self.calculate_WMA(data["loss"], period_WMA))
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0, self.calculate_WMA(data["gain"], period_WMA), sum)
        data[f"RSI-WMA-{period_WMA}"] = 100 * self.calculate_WMA(data["gain"], period_WMA) / avg_sum["avg_sum"]
        return data[f"RSI-WMA-{period_WMA}"]
    
    def calculate_RSI_HMA(self, data_HMA, period_HMA):
        data = data_HMA.copy()
        sum = (self.calculate_HMA(data["gain"], period_HMA) + self.calculate_HMA(data["loss"], period_HMA))
        avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
        avg_sum["avg_sum"] = np.where(sum == 0, self.calculate_HMA(data["gain"], period_HMA), sum)
        data[f"RSI-HMA-{period_HMA}"] = 100 * self.calculate_HMA(data["gain"], period_HMA) / avg_sum["avg_sum"]     
        return data[f"RSI-HMA-{period_HMA}"]
    
    # Get RSI data
    def get_RSI_data(self):
        data = self.data.copy()
        if self.RSI_type == "SMA":
            data[f"RSI-SMA-{self.RSI_period}"] = self.calculate_RSI_SMA(self.prepared_data, self.RSI_period)
            self.results_RSI[f"RSI-SMA-{self.RSI_period}"] = data[f"RSI-SMA-{self.RSI_period}"]
            #self.results_RSI.dropna(inplace = True)
            self.results_RSI = self.results_RSI.dropna()
        elif self.RSI_type == "EMA":
            data[f"RSI-EMA-{self.RSI_period}"] = self.calculate_RSI_EMA(self.prepared_data, self.RSI_period)
            self.results_RSI[f"RSI-EMA-{self.RSI_period}"] = data[f"RSI-EMA-{self.RSI_period}"]
            #self.results_RSI.dropna(inplace = True)
            self.results_RSI = self.results_RSI.dropna()
        elif self.RSI_type == "WMA":
            data[f"RSI-WMA-{self.RSI_period}"] = self.calculate_RSI_WMA(self.prepared_data, self.RSI_period)
            self.results_RSI[f"RSI-WMA-{self.RSI_period}"] = data[f"RSI-WMA-{self.RSI_period}"]
            #self.results_RSI.dropna(inplace = True)
            self.results_RSI = self.results_RSI.dropna()
        elif self.RSI_type == "HMA":
            data[f"RSI-HMA-{self.RSI_period}"] = self.calculate_RSI_HMA(self.prepared_data, self.RSI_period)
            self.results_RSI[f"RSI-HMA-{self.RSI_period}"] = data[f"RSI-HMA-{self.RSI_period}"]
            #self.results_RSI.dropna(inplace = True)
            self.results_RSI = self.results_RSI.dropna()
        elif self.RSI_type == "MA":
            data[f"RSI-SMA-{self.RSI_period}"] = self.calculate_RSI_SMA(self.prepared_data, self.RSI_period)
            data[f"RSI-EMA-{self.RSI_period}"] = self.calculate_RSI_EMA(self.prepared_data, self.RSI_period)
            data[f"RSI-WMA-{self.RSI_period}"] = self.calculate_RSI_WMA(self.prepared_data, self.RSI_period)
            data[f"RSI-HMA-{self.RSI_period}"] = self.calculate_RSI_HMA(self.prepared_data, self.RSI_period)
            #self.results = data.drop(["Close"], inplace = True).dropna(inplace = True)
            self.results_RSI = data.copy()
            #self.results_RSI.drop(columns = ["Close"], inplace = True)
            self.results_RSI = self.results_RSI.drop(columns = ["Close"])
            #self.results_RSI.dropna(inplace = True)
            self.results_RSI = self.results_RSI.dropna()
        #self.data = data
        #return self.data.dropna(inplace = True)   
   
    #Averages
    def calculate_WMA(self, data_WMA, period_WMA):
        data = data_WMA.copy()
        data["WMA"] = data.rolling(period_WMA).apply(lambda x: ((np.arange(period_WMA)+1)*x).sum()/(np.arange(period_WMA)+1).sum(), raw=True)
        return data["WMA"]

    def calculate_HMA(self, data_HMA, period_HMA):        
        data = data_HMA.copy()
        a = self.calculate_WMA(data.copy(), period_HMA//2).multiply(2)
        b = self.calculate_WMA(data.copy(), period_HMA)
        c = a - b
        data["HMA"] = self.calculate_WMA(c, int(np.sqrt(period_HMA)))
        return data["HMA"]
    
    # Stochastic calculation
    def calculate_stochastic(self):
        #SO K line:
        if self.RSI_type == "SMA":
            min = self.results_RSI[f"RSI-SMA-{self.RSI_period}"].rolling(self.K_period).min()
            max = self.results_RSI[f"RSI-SMA-{self.RSI_period}"].rolling(self.K_period).max()
            #self.K = ((self.results_RSI[f"RSI-SMA-{self.RSI_period}"] - min) / (max - min))*100

            sum = (max - min)
            avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
            avg_sum["avg_sum"] = np.where(sum == 0, (self.results_RSI[f"RSI-SMA-{self.RSI_period}"] - min), sum)
            self.K = 100 * (self.results_RSI[f"RSI-SMA-{self.RSI_period}"] - min) / avg_sum["avg_sum"]

            self.results_SO_RSI[f"SORSI-SMA-%K-{self.K_period}"] = self.K
            #self.results_SO_RSI.dropna(inplace = True)

        elif self.RSI_type == "EMA":
            min = self.results_RSI[f"RSI-EMA-{self.RSI_period}"].rolling(self.K_period).min()
            max = self.results_RSI[f"RSI-EMA-{self.RSI_period}"].rolling(self.K_period).max()
            #self.K = ((self.results_RSI[f"RSI-EMA-{self.RSI_period}"] - min) / (max - min))*100
            
            sum = (max - min)
            avg_sum = pd.DataFrame(sum, index=sum.index, columns=["avg_sum"]) 
            avg_sum["avg_sum"] = np.where(sum == 0, (self.results_RSI[f"RSI-EMA-{self.RSI_period}"] - min), sum)
            self.K = 100 * (self.results_RSI[f"RSI-EMA-{self.RSI_period}"] - min) / avg_sum["avg_sum"]           
            
            self.results_SO_RSI[f"SORSI-EMA-%K-{self.K_period}"] = self.K
            #self.results_SO_RSI.dropna(inplace = True)

        elif self.RSI_type == "HMA":
            min = self.results_RSI[f"RSI-HMA-{self.RSI_period}"].rolling(self.K_period).min()
            max = self.results_RSI[f"RSI-HMA-{self.RSI_period}"].rolling(self.K_period).max()
            self.K = ((self.results_RSI[f"RSI-HMA-{self.RSI_period}"] - min) / (max - min))*100
            self.results_SO_RSI[f"SORSI-HMA-%K-{self.K_period}"] = self.K
            #self.results_SO_RSI.dropna(inplace = True)

        #SO D line from K:
        if self.D_smoothing_type == "SMA":
            self.D = self.K.rolling(self.D_period).mean()
            self.results_SO_RSI[f"SORSI-SMA-%D-{self.D_period}"] = self.D
            #self.results_SO_RSI.dropna(inplace = True)
        elif self.D_smoothing_type == "EMA":
            self.D = self.K.ewm(span = self.D_period, adjust=False, min_periods = self.EMA_min_periods).mean()
            self.results_SO_RSI[f"SORSI-EMA-%D-{self.D_period}"] = self.D
            #self.results_SO_RSI.dropna(inplace = True)
        elif self.D_smoothing_type == "HMA":
            self.D = self.calculate_HMA(self.K, self.D_period)
            self.results_SO_RSI[f"SORSI-HMA-%D-{self.D_period}"] = self.D
            #self.results_SO_RSI.dropna(inplace = True)
        
        #SO D slow line:
        if self.D_smoothing_type == "SMA":
            self.D_slow = self.D.rolling(self.D_slow_period).mean()
            self.results_SO_RSI[f"SORSI-SMA-%D_slow-{self.D_slow_period}"] = self.D_slow
            #self.results_SO_RSI.dropna(inplace = True)
        elif self.D_smoothing_type == "EMA":
            self.D_slow = self.D.ewm(span = self.D_slow_period, adjust=False, min_periods = self.EMA_min_periods).mean()
            self.results_SO_RSI[f"SORSI-EMA-%D_slow-{self.D_slow_period}"] = self.D_slow
        elif self.D_smoothing_type == "HMA" and self.D_slow_period > 1:
            self.D_slow = self.calculate_HMA(self.D, self.D_slow_period)
            self.results_SO_RSI[f"SORSI-HMA-%D_slow-{self.D_slow_period}"] = self.D_slow
            #self.results_SO_RSI.dropna(inplace = True)
        
        #self.results_SO_RSI.dropna(inplace = True)
        #self.results_SO_RSI["Date"] =  self.results_SO_RSI.index
    
    def plot_SO_RSI(self):
        self.results_SO_RSI.plot(figsize=(12, 8))
        
# CrossOver
class CrossOver():
    def __init__(self, data0 , data1):
        self.data0 = data0
        self.data1 = data1
        self.diff = pd.DataFrame()
        self.upcross = pd.DataFrame()
        self.downcross = pd.DataFrame()
        self.crossover = pd.DataFrame()
        self.crossover = pd.DataFrame()
        self.calculate_data()
        
    def __repr__(self):
        return "CrossOver(data0 = {}, data1 = {})".format(self.data0, self.data1)
    
    def calculate_data(self):

        if isinstance(self.data1, int) or isinstance(self.data1, float):
            x = self.data1
            self.data1 = self.data0.copy()
            self.data1.loc[:] = x
        
        data0 = self.data0.copy()
        data1 = self.data1.copy()

        name0 = data0.columns.values.tolist()
        name0 = name0[0]

        name1 = data1.columns.values.tolist()
        name1 = name1[0]
        
        data0.rename(columns = {name0:"data0"}, inplace = True)
        data1.rename(columns = {name1:"data1"}, inplace = True)

        self.diff["diff"] = self.data0 - self.data1
        
        for i in range(len(self.diff["diff"])-1):
            if self.diff["diff"].iloc[i] < 0 and data0["data0"].iloc[i+1] > data1["data1"].iloc[i+1]:
                self.upcross.at[i, "up"] = 1
                self.downcross.at[i,"down"] = 0 
            elif self.diff["diff"].iloc[i] > 0 and data0["data0"].iloc[i+1] < data1["data1"].iloc[i+1]:
                self.downcross.at[i, "down"] = -1
                self.upcross.at[i, "up"] = 0
            else:
                self.upcross.at[i, "up"] = 0
                self.downcross.at[i,"down"] = 0
                
            self.crossover.at[i, "cross"] = self.upcross.at[i, "up"] + self.downcross.at[i, "down"]

        self.crossover["date"] = self.diff.index.drop(self.diff.index[0])
        self.crossover.set_index(keys ='date', inplace = True)
