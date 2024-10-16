from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib # type: ignore

import tensorflow.compat.v2.feature_column as fc # type: ignore
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

TF_ENABLE_ONEDNN_OPTS=0

train_url = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
eval_url =  'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'

column_names = ['survived', 'sex','age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

train_set = pd.read_csv(train_url, names=column_names,
                          na_values='?', comment='\t',
                          sep=',', skipinitialspace=True) # training data
eval_set = pd.read_csv(eval_url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True) # testing data
# print(dftrain.head()) gives first 5 rows
# print(dftrain.describe()) 5 item summary thingy from statistics
# y_train = dftrain.pop('survived')
# y_eval = dfeval.pop('survived')

# print(train_set['embark_town'].unique())
dataset = train_set.copy()
# print(dataset.tail())
# print(dataset.isna().sum())
dataset.dropna()

# dataset['sex'] = dataset['sex'].map({'male': 1, 'female': 2})
# dataset['class'] = dataset['class'].map({'First': 1, 'Second': 2, 'Third': 3})
# dataset['deck'] = dataset['deck'].map({'unknown': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8})
# dataset['embark_town'] = dataset['embark_town'].map({'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3, 'unknown': 4})
# dataset['alone'] = dataset['alone'].map({'y': 1, 'n': 2})

# dataset = pd.get_dummies(dataset, columns=['sex', 'class', 'deck', 'embark_town', 'alone'], prefix='', prefix_sep='')
# print(dataset.tail())
dataset = pd.get_dummies(dataset, columns=['sex', 'class', 'deck', 'embark_town', 'alone'], prefix='', prefix_sep='', dtype='int')
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.dropna()

# if isinstance(dataset.iloc[0, 0], str):  # Check if the first row contains strings (headers)
#     dataset = dataset.iloc[1:] 
# if isinstance(dataset.iloc[0, 0], str):  # Check if the first row contains strings (headers)
#     dataset = dataset.iloc[1:] 

train_features = dataset.copy()
test_features = eval_set.copy()

train_labels = train_features.pop('survived')
test_labels = test_features.pop('survived')


# train_set.describe().transpose()[['mean', 'std']]
print(train_features.head())



# train_features_np = np.array(train_features)

#normalize the data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


# Plain Lin REg