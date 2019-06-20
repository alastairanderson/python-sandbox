# https://github.com/quandl/quandl-python
# https://pypi.org/project/Quandl/

# requirements.txt
# quandl==3.4.8

import quandl


# TODO: Ensure these are stored in environment variables before checking this in.
API_KEY = ''


# 1. Authenticate
quandl.ApiConfig.api_key = API_KEY

# Example: Download data for Nokia
data = quandl.get_table('MER/F1', compnumber="39102", paginate=True)

# Notes:
#  1. type(data) = pandas DataFrame
