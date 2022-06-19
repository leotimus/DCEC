from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K

from reader.DataGenerator import DataGenerator

def CAE(filters=[32, 64, 128, 60, 256], contig_len=20000):
    input_shape = (contig_len, 4)
    model = keras.models.Sequential()
    #model.add(keras.layers.Masking(mask_value=-1., input_shape=input_shape))
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(keras.layers.Conv1D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    #model.add(keras.layers.LeakyRelu(alpha=0.3))
    model.add(keras.layers.Conv1D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(keras.layers.Conv1D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    
    #model.add(keras.layers.Conv1D(filters[4], 3, strides=2, padding=pad3, activation='relu', name='conv4'))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=filters[3], name='embedding'))
    model.add(keras.layers.Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[1]/4), activation='relu'))

    model.add(keras.layers.Reshape((int(input_shape[0]/8), filters[2])))

    #model.add(keras.layers.Conv1DTranspose(filters[2], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(keras.layers.Conv1DTranspose(filters[1], 3, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(keras.layers.Conv1DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv1'))
    model.add(keras.layers.Conv1DTranspose(input_shape[1], 5, strides=2, padding='same', activation='softmax', name='deconv4'))
    model.summary()
    return model



