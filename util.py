import tensorflow as tf
import numpy as np

#-------------------------------------------------------
# normalize dataframe between 0 and 1
#-------------------------------------------------------
def normalize(dataframe):
    return (dataframe - dataframe.values.min()) / (dataframe.values.max() - dataframe.values.min())

#-------------------------------------------------------
# convert pd dataframe to tf dataset
#   note that if working with a large datafile,
#   should use tf.data to read csv from disk directly
#-------------------------------------------------------
def df_to_dataset(dataframe, label_col_name=None, shuffle=True, batch_size=None):

    if label_col_name:
        # make copy of df
        tmp_dataframe = dataframe.copy()

        # remove labels from df
        labels = tmp_dataframe.pop(label_col_name)

        # normalize df
        tmp_dataframe = normalize(tmp_dataframe)

        x = tf.data.Dataset.from_tensor_slices(tmp_dataframe.values)
        y = tf.data.Dataset.from_tensor_slices(labels).map(lambda z: tf.one_hot(z, 10))
        
        dataset = tf.data.Dataset.zip((x, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dataframe.values)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(tmp_dataframe))

    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset