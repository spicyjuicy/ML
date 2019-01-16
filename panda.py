import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class Stock:

    def __init__(self, file,split_size):
        self.file = file
        self.ready_up()
        self.split(split_size)

    def ready_up(self):
        
        df = pd.read_csv(self.file)

        df['Day_Name'] = pd.to_datetime(df['Date'])
        #print(df['Date'].iloc[0],df['Date'].iloc[-1])
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

        df = df.drop(columns=['Date','Day_Name','DayNum','YearNum','rank'])

        for x in range(5):
            if df.major_index.iloc[-1] != 5:
                df = df[:-1]

        for column in df:
            df[column] = df[column]/df[column].max()

        df = df.drop(columns=['major_index','WeekNum'])

        self.df2 = df

    def split(self,split_size):
        #print(len(self.df2))

        self.arrs = np.split(self.df2.values,len(self.df2)/split_size)        
        

    def print_head(self,size):

        print(self.df2.head(size))
        print(self.df2.dtypes,len(self.df2.columns),len(self.df2))

    def print_tail(self,size):

        print(self.df2.tail(size))
        print(self.df2.dtypes,len(self.df2.columns))

   
def create_labels(array):

    labels = np.zeros(len(array),dtype=int)
    for x in range(len(array)-1):
        if array[x+1][0][3] > array[x][4][3]:
            labels[x] = int(0)
        else:
            labels[x] = int(1)

    return labels

def split_data(full_data,full_labels, size):

    arr1 , arr2 = full_data[:size] , full_data[size:]
    lab1 , lab2 = full_labels[:size] , full_labels[size:]

    return (arr1 ,lab1), (arr2,lab2)






df = Stock('FB.csv',5)
print(len(df.df2))

#print(df.arrs)
print(np.shape(df.arrs))
create_labels(df.arrs)
df.print_head(20)
df.print_tail(20)
#df.print_tail(20)

arrays = df.arrs
labels = create_labels(arrays)

class_names = ['Up','Down']


(train_images,train_labels) , (test_images,test_labels) = split_data(arrays,labels,1000)
print(np.shape(test_images))

plt.figure(figsize=(5,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels[i])])
#plt.show()

print(np.shape(test_images),np.shape(test_images),test_labels)

test_images = np.asarray(test_images)
print(type(test_images))


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5, 6)),
    keras.layers.Dense(5, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(train_images), np.array(train_labels), epochs=5)

test_loss, test_acc = model.evaluate(np.array(test_images), np.array(test_labels))

print('Test accuracy:', test_acc)