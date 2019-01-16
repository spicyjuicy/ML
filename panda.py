import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class Stock:

    def __init__(self, file,split_size):
        self.file = file
        self.split_size = split_size
        self.ready_up()
        self.split()
        self.create_labels()
        self.split_data()

    def ready_up(self):
        
        df = pd.read_csv(self.file)

        df['Day_Name'] = pd.to_datetime(df['Date'])
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

        self.dataframe = df

    def split(self):

        self.split_array = np.split(self.dataframe.values,len(self.dataframe)/5)
       # arr = self.split_array
      #  self.split_array = np.random.shuffle(arr)

        

    def print_head(self,size):

        print(self.dataframe.head(size))
        
    def print_tail(self,size):

        print(self.df2.tail(size))

    def print_stats(self):
        print(self.dataframe.dtypes,len(self.dataframe.columns),len(self.dataframe))
        
    def create_labels(self):

        labels = np.zeros(len(self.split_array),dtype=int)
        for x in range(len(self.split_array)-1):
            if self.split_array[x+1][0][3] > self.split_array[x][4][3]:
                labels[x] = int(0)
            else:
                labels[x] = int(1)

        self.labels_array = labels

    def split_data(self):

        size = int(len(self.split_array) / 2)

        self.train_data , self.test_data = self.split_array[:size] , self.split_array[size:]
        self.train_labels , self.test_labels = self.labels_array[:size] , self.labels_array[size:]






FB = Stock('FB.csv',5)




class_names = ['Up','Down']

train_images = FB.train_data
train_labels = FB.train_labels

test_images = FB.test_data
test_labels = FB.test_labels


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