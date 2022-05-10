from tensorflow import keras
from keras.models import Model, Sequential


def create_auto_encoder_models2(dims=[20000, 10000, 5000, 1000], act='relu', init='glorot_uniform') -> Model:
    input_shape = (20000, 4)
    n_stacks = len(dims) - 1
    # input
    x = keras.layers.Input(shape=20000, name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks - 1):
        h = keras.layers.Dense(dims[i + 1], activation=act, kernel_initializer=init, name=f'encoder_{i}')(h)

    # hidden layer
    h = keras.layers.Dense(dims[-1], kernel_initializer=init, name=f'encoder_{(n_stacks - 1)}')(
        h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        y = keras.layers.Dense(dims[i], activation=act, kernel_initializer=init, name=f'decoder_{i}')(y)

    # output
    y = keras.layers.Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


def create_auto_encoder_models(act='relu', init='glorot_uniform') -> Sequential:
    bp_len = 1000
    input_shape = (bp_len, 4)
    model: keras.models.Sequential = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape, name='input'))
    model.add(keras.layers.Masking(mask_value=-1., input_shape=input_shape))
    model.add(keras.layers.Dense(512, activation=act, kernel_initializer=init, name='encoder_0'))
    model.add(keras.layers.Dense(256, activation=act, kernel_initializer=init, name='encoder_1'))
    model.add(keras.layers.Dense(128, activation=act, kernel_initializer=init, name='encoder_2'))
    model.add(keras.layers.Dense(64, activation=act, kernel_initializer=init, name='encoder_3'))
    model.add(keras.layers.Dense(128, activation=act, kernel_initializer=init, name='decoder_3'))
    model.add(keras.layers.Dense(256, activation=act, kernel_initializer=init, name='decoder_2'))
    model.add(keras.layers.Dense(512, activation=act, kernel_initializer=init, name='decoder_1'))
    model.add(keras.layers.Dense(4, kernel_initializer=init, name='output'))

    return model
