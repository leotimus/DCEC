from keras import Model, Input
from keras.layers import Dense
from keras.losses import mse
from tensorflow import keras


class SAE(object):

    def __init__(self, batch_size=100, n_epoch=100, n_hidden=256, input_shape=None,
                 print_model=True, save_dir='results/', loss_func=None):
        super(SAE, self).__init__()
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.print_model = print_model
        self.save_dir = save_dir

  
        self.encoder_input, self.encoder = self.create_encoder(input_shape)
        self.decoder = self.decoder_model(input_shape)
        self.outputs = self.decoder(self.encoder(self.encoder_input))

        self.model = Model(self.encoder_input, self.outputs, name='sae')
        if self.print_model:
            self.model.summary()
            keras.utils.plot_model(self.model, to_file=f'{self.save_dir}/sae.png', show_shapes=True)

    def create_encoder(self, input_shape) -> (Input, Model, Dense, Dense):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.n_hidden, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        y = Dense((self.n_hidden / 2), activation='relu')(x)
        y = keras.layers.BatchNormalization()(y)
        output = Dense((self.n_hidden / 4), name='output')(y)
        output = keras.layers.BatchNormalization()(output)
        encoder = Model(inputs, output, name='encoder')
        if self.print_model:
            encoder.summary()
            keras.utils.plot_model(encoder, to_file=f'{self.save_dir}/sae_encoder.png', show_shapes=True)
        return inputs, encoder

    def mse_loss(self, encoder_input, outputs):
        ms_error = mse(encoder_input, outputs)
        return ms_error


    def decoder_model(self, input_shape):
        latent_inputs = Input(shape=(int(self.n_hidden / 4),), name='z')
        x = Dense((self.n_hidden /2), activation='relu')(latent_inputs)
        x = keras.layers.BatchNormalization()(x)
        y = Dense(self.n_hidden, activation='relu')(x)
        y = keras.layers.BatchNormalization()(y)
        outputs = Dense(input_shape[0], activation='sigmoid')(y)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        if self.print_model:
            decoder.summary()
            keras.utils.plot_model(decoder, to_file=f'{self.save_dir}/sae_decoder.png', show_shapes=True)
        return decoder