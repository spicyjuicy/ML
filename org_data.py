import csv
import numpy as np

def invalid(row):

    for x in row:
        if x == 'Nan' or x == 'nan' or x == 'NaN':
            return True
    
    return False


def list_to_array(row):

    arr = np.array([])

    for x in row:
       # print(x,type(x),float(x))
        arr = np.append(arr,float(x))

    return arr

    
 

datafilename = 'bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv'

d = ','
f=open(datafilename,'r')

reader=csv.reader(f,delimiter=d)
ncol=len(next(reader)) # Read first line and count columns

next(reader)


block = np.array([])
data = np.array([])


y = 0

for row in reader:

    if invalid(row):
        continue

    row = list_to_array(row)
    y = y + 1
    
    block = np.append(block,row)
    if y == 10:        
        block = np.reshape(block,(-1,ncol))
        print(block)
        print(block.shape)
        data = np.append(data,block)
        print(data.shape)
        break
        
    

