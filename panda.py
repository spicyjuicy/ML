import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Stock:
    
    def __init__(self, file):
        self.file = file
        self.df2 = pd.read_csv(self.file)

    def ready_up(self):
        df = self.df2
       # df = pd.read_csv(self.file)

        df['Day_Name'] = pd.to_datetime(df['Date'])
        print(df['Date'].iloc[0],df['Date'].iloc[-1])
        idx = pd.date_range(df['Date'].iloc[0],df['Date'].iloc[-1])

        df.index = pd.DatetimeIndex(df['Day_Name'])

        df = df.reindex(idx,fill_value=0.0)

        df['Date'] = df.index
        df = df.reset_index(drop=True)
        df['Day_Name'] = pd.to_datetime(df['Date'])
        df['DayNum'] = df['Day_Name'].dt.day
        df['WeekNum'] = df['Day_Name'].dt.week
        df['YearNum'] = df['Day_Name'].dt.year


        df = df[df.WeekNum != df.WeekNum.iloc[0]]
    
        df['major_index'] = (df.index -2) % 7
        df['rank'] = df.groupby(['YearNum'])['Date'].rank()
        df = df[df.major_index != 6]
        df = df[df.major_index != 0]

        df = df.drop(columns=['Day_Name','DayNum','WeekNum','YearNum','rank'])

        for x in range(5):
            if df.major_index.iloc[-1] != 5:
                df = df[:-1]

       # df['Next Close'] = df.Close.shift(-4)

        self.df2 = df

    def print_head(self,size):

        print(self.df2.head(size))
        print(self.df2.dtypes,len(self.df2.columns),len(self.df2))

    def print_tail(self,size):

        print(self.df2.tail(size))
        print(self.df2.dtypes,len(self.df2.columns))

        


df = Stock('FB.csv')
df.ready_up()
df.print_head(20)
#df.print_tail(20)
#yoyoyo