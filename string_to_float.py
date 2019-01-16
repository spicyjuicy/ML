import csv
import numpy as np

datafilename = 'bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv'


d = ','
f=open(datafilename,'r')

reader=csv.reader(f,delimiter=d)

result_file = "clean.csv" 

f = open(result_file,'w')

next(reader)

for row in reader:
   # next(reader)

    row2 = str(row)
    row2 = row2.replace("[","")
    row2 = row2.replace("]","")
    
    if row2[0] != 'NaN':
        for x in row:
            print(type(x),float(x))
            f.write(float(x))
        f.write("\n")
    

f.close()
