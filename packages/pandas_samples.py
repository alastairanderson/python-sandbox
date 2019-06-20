'''requirements.txt
pandas==0.24.2
xlrd==1.2.0

VSCode running notes:
Open in the 'packages' folder and venv created in their also.
'''

import pandas as pd


excel_file = pd.ExcelFile("./pandas_sample_data.xlsx")

# http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
df = pd.read_excel(excel_file, "Sheet1", usecols=None)

print("\nShape")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html
print(df.shape)

print("\nHead")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html
print(df.head(3))

print("\nTail")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tail.html
print(df.tail(3))

print("\nDescribe")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
print(df.describe())

print("\nMean")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
print(df.mean())

print("\nStandard Deviation")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html
print(df.std())

print("\nNormalising an entire column")
df_mean = df.mean()
df_std = df.std()
df['Average Rating'] = (df['Average Rating'] - df_mean.get('Average Rating'))/df_std.get('Average Rating')
print(df.head(5))

print("\nOne-hot encode values")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
df = pd.get_dummies(df, columns=['Sex'])
print(df.head(3))

print("\nDrop rows")
print(df.head(3))
df = df[2:]
print(df.head(3))

print("\nDrop columns")
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
print(df.head(3))
df = df.drop('Description', 1)
print(df.head(3))

print("\nAdding a dictionary record")
print("\n1. Create new dictionary")
record = {}
record["Id"] = 10
record["Name"] = "New name"
record["Date"] = "2019-06-16"       # this is not a date
record["Author"] = "Rachel Do"
record["Average Rating"] = 1.2
record["Sex_Female"] = 1
record["Sex_Male"] = 0

print("\n2. Convert dictionary to pandas Series")
s = pd.Series(record)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_frame.html

print("\n3. Convert series to dataframe and concatenate")
df = pd.concat([df, s.to_frame().T])
print(df.tail(3))
