from __future__ import absolute_import, division, print_function
import pandas as pd
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
    
df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv')


df = df.dropna()
df['Name'] = df['Timestamp'].shift(1) - df['Timestamp']
df['Forward'] = df['Close'].shift(-1)

print(df.head(10))

df = df.dropna()
arr = df.values


def build_model():
    model = keras.Sequential([
        layers.Dense(72, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
        ])
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

column_names = ['Timestamp','Open','High','Low','Close',
                 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price'] 
raw_dataset = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv')#, names=column_names,
                     # na_values = "?", comment='\t',
                      #sep=" ", skipinitialspace=True)

dataset = raw_dataset.dropna()
dataset['Name'] = dataset['Timestamp'].shift(1) - dataset['Timestamp']
dataset['Forward'] = dataset['Close'].shift(-1)

'''
raw_dataset = raw_dataset[['Forward','Timestamp','Open','High','Low','Close',
                 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price','Name']]
                 '''
print(dataset.head(10))


dataset.tail()

dataset.isna().sum()

dataset = dataset.dropna()


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Forward")
train_stats = train_stats.transpose()

print(train_stats)

train_labels = train_dataset.pop('Forward')
test_labels = test_dataset.pop('Forward')

def norm(x):
      return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)



model = build_model()

model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 10

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Forward]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Forward^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])

plot_history(history)
plt.show()

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Forward".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Forward]')
plt.ylabel('Predictions [Forward]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
