import numpy as np
import pandas as pd

# the following replaces: pandas.io.data
import pandas_datareader.data as web

import datetime

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2013, 1, 27)

# f = web.DataReader('F', 'google', start, end)
# f = web.DataReader('F', 'morningstar', start, end)
# f.head()



# QUANDL API Token required for this to work
symbol = 'WIKI/AAPL'  # or 'AAPL.US'
df = web.DataReader(symbol, 'quandl', '2015-01-01', '2015-01-05')
df.loc['2015-01-02']



