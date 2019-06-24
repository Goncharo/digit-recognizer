from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np

from util import df_to_dataset
from sklearn.model_selection import train_test_split



#----------------------------------------
# data prep
#----------------------------------------

# header: label, pixel0 ... pixel784
# size: 42,000 x 785
TRAIN_FILE_NAME = './train.csv'

# header: pixel0 ... pixel784
# size: 28,000 x 784
TEST_FILE_NAME = './test.csv'

# read train and test csvs into dataframes
TRAIN_DATAFRAME = pd.read_csv(TRAIN_FILE_NAME)
TEST_DATAFRAME = pd.read_csv(TEST_FILE_NAME)

# split training data into train (90%) and dev (10%) set
TRAIN_DATAFRAME, DEV_DATAFRAME = train_test_split(TRAIN_DATAFRAME, test_size=0.1)

batch_size = 32
train = df_to_dataset(TRAIN_DATAFRAME, label_col_name='label', batch_size=batch_size)
dev = df_to_dataset(DEV_DATAFRAME, label_col_name='label', shuffle=False, batch_size=batch_size)
test = df_to_dataset(TEST_DATAFRAME, shuffle=False, batch_size=1)



#----------------------------------------
# model compilation
#----------------------------------------

# create basic network with 10 outputs, for each digit
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit model to train data set, use dev set for validation, train for 10 epochs
model.fit(train, validation_data=dev, epochs=10)



#----------------------------------------
# output predictions
#----------------------------------------

# get array of predictions for each test input
predictions = model.predict(test)

# create submission file containing predictions using the format:
# ImageId,Label
# 1,0
# 2,0
submission_file = open("submission.csv", "w+")
submission_file.write("ImageId,Label")
for index, pred in enumerate(predictions):
    submission_file.write("\n")
    submission_file.write("{},{}".format(index + 1, np.argmax(pred)))
submission_file.close()
