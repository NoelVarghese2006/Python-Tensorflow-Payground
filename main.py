import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers # type: ignore

np.set_printoptions(precision=3, suppress=True)

train_url = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
eval_url = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'

column_names = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

# Load data
train_set = pd.read_csv(train_url, names=column_names, na_values='?', comment='\t', sep=',', skipinitialspace=True)
eval_set = pd.read_csv(eval_url, names=column_names, na_values='?', comment='\t', sep=',', skipinitialspace=True)

# Convert necessary columns to numeric
numeric_columns = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
for col in numeric_columns:
    train_set[col] = pd.to_numeric(train_set[col], errors='coerce')
    eval_set[col] = pd.to_numeric(eval_set[col], errors='coerce')

# Drop missing values
train_set = train_set.dropna()
eval_set = eval_set.dropna()

# Create dummy variables for categorical columns
train_set = pd.get_dummies(train_set, columns=['sex', 'class', 'deck', 'embark_town', 'alone'], prefix='', prefix_sep='')
eval_set = pd.get_dummies(eval_set, columns=['sex', 'class', 'deck', 'embark_town', 'alone'], prefix='', prefix_sep='')

# Ensure both train and eval sets have the same columns
train_set, eval_set = train_set.align(eval_set, join='left', axis=1, fill_value=0)

# Separate features and labels
train_features = train_set.copy()
eval_features = eval_set.copy()

train_label = train_features.pop('survived')
eval_label = eval_features.pop('survived')

# Convert features to float
train_features = train_features.astype('float32')
eval_features = eval_features.astype('float32')

print(train_set['survived'].value_counts())

# Normalize the data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# Build and compile the model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    return model

# Train the DNN model
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_label,
    validation_split=0.2,
    verbose=0, epochs=100
)

# Evaluate and predict
test_results = dnn_model.evaluate(eval_features, eval_label, verbose=0)

print(f"Test loss: {test_results[0]}")
print(f"Test accuracy: {test_results[1]}")

test_predictions = dnn_model.predict(eval_features).flatten()

# Print results
# print(eval_label)
# print(test_predictions)
