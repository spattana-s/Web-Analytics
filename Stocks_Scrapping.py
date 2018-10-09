
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
# date time to use date objects
from datetime import date
import pandas as pd



def stockdata(stocksymbol):
        
    print(stocksymbol)
    start = date(2018, 1, 10) #yyyymmdd
    end = date(2018, 3, 15)
    #output_file="C:/Users/shrey/OneDrive/Desktop/seaktop/660 Web/Project/stock_raw_data/"+stocksymbol+".csv"
    stock = DataReader(stocksymbol, 'google', start, end)
    my_df = pd.DataFrame(stock)
    my_df.to_csv("C:/Users/shrey/OneDrive/Desktop/seaktop/660 Web/Project/"+stocksymbol+"_stock.csv")

    return stock    
        
if __name__ == "__main__": 
    List_stock_symbol = ["FB","BAC","BA","AMZN","GOOGL","AAPL", "JPM","NFLX", "AABA","MSFT"]
    for symbol in List_stock_symbol:
        print (symbol)
        stockdata(symbol)
        
    

