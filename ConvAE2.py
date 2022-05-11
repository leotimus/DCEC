from tensorflow import keras
import numpy as np

from reader.DataGenerator import DataGenerator


class CAE2(keras.models.Sequential):

    def __init__(self, filters=[32, 64, 128, 60], contig_len=20000):
        super(CAE2, self).__init__()

        input_shape = (contig_len, 4)
        # model = keras.models.Sequential()
        self.add(keras.layers.Masking(mask_value=-1., input_shape=input_shape))
        if input_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'

        self.add(keras.layers.Conv1D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1',
                                     input_shape=input_shape))

        self.add(keras.layers.Conv1D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

        self.add(keras.layers.Conv1D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(units=filters[3], name='embedding'))
        self.add(
            keras.layers.Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[1] / 4), activation='relu'))

        self.add(keras.layers.Reshape((int(input_shape[0] / 8), filters[2])))
        # model.summary()
        self.add(
            keras.layers.Conv1DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

        self.add(
            keras.layers.Conv1DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
        # model.summary()
        self.add(keras.layers.Conv1DTranspose(input_shape[1], 5, strides=2, padding='same', name='deconv1'))
        self.summary()

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        predict = super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers,
                                  use_multiprocessing)
        print(f'the cae\'s prediction is {predict}')
        return predict