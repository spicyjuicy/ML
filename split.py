import csv
import numpy as np

datafilename = 'bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv'
split_len = 1000000


d = ','
f=open(datafilename,'r')

reader=csv.reader(f,delimiter=d)
ncol=len(next(reader)) # Read first line and count columns
print(ncol)

x = 0
y = 0
z = 0

result_file = "split_0.csv" 

f = open(result_file,'w')

stop = split_len
for row in reader:
 
        
   # break
    row2 = str(row)
    row2 = row2.replace("[","")
    row2 = row2.replace("]","")
    
    
    if row[1] != 'NaN':
        f.write(row2 + "\n")
        y = y + 1
   
    if(y == stop):
       
        f.close()
        z = z + 1
        stop = stop + split_len
        result_file = "split_" + str(z) + ".csv"
        print(z,result_file)
       
        f = open(result_file,'w')

f.close()

