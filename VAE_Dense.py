from keras import Model, backend as K, Input
from keras.layers import Dense, Lambda
from keras.losses import mse
import tensorflow as tf
from tensorflow import keras


class VAE(object):

    def __init__(self, batch_size=100, n_epoch=100, n_hidden=256, input_shape=None,
                 print_model=True, save_dir='results/', loss_func=None):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.print_model = print_model
        self.save_dir = save_dir

  
        self.encoder_input, self.encoder = self.create_encoder(input_shape)
        self.decoder = self.decoder_model(input_shape)
        self.outputs = self.decoder(self.encoder(self.encoder_input))

        self.model = Model(self.encoder_input, self.outputs, name='vae')
        if self.print_model:
            self.model.summary()
            keras.utils.plot_model(self.model, to_file=f'{self.save_dir}/vae.png', show_shapes=True)

    def create_encoder(self, input_shape) -> (Input, Model, Dense, Dense):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.n_hidden, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        output2 = Dense((self.n_hidden / 2), name='output2')(x)
        output2 = keras.layers.BatchNormalization()(output2)
        output = Dense((self.n_hidden / 4), name='output')(output2)
        output = keras.layers.BatchNormalization()(output)
        encoder = Model(inputs, output, name='encoder')
        if self.print_model:
            encoder.summary()
            keras.utils.plot_model(encoder, to_file=f'{self.save_dir}/vae_encoder.png', show_shapes=True)
        return inputs, encoder

    def mse_loss(self, encoder_input, outputs):
        ms_error = mse(encoder_input, outputs)
        return ms_error


    def decoder_model(self, input_shape):
        latent_inputs = Input(shape=(int(self.n_hidden / 4),), name='z')
        x2 = Dense(self.n_hidden / 2, activation='relu')(latent_inputs)
        x2 = keras.layers.BatchNormalization()(x2)
        x = Dense(self.n_hidden, activation='relu')(x2)
        x = keras.layers.BatchNormalization()(x)
        outputs = Dense(input_shape[0], activation='sigmoid')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        if self.print_model:
            decoder.summary()
            keras.utils.plot_model(decoder, to_file=f'{self.save_dir}/vae_decoder.png', show_shapes=True)
        return decoder