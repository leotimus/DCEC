from keras import Model, backend as K, Input
from keras.layers import Dense, Lambda
from keras.losses import mse
from tensorflow import keras

class VAE(object):

    def __init__(self, batch_size=100, n_epoch=100, n_hidden=256, input_shape=None,
                 print_model=True, save_dir='results/'):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.print_model = print_model
        self.save_dir = save_dir

        self.encoder_input, self.encoder, z_mean, z_log_var = self.create_encoder(input_shape)
        self.decoder = self.decoder_model(input_shape)
        self.outputs = self.decoder(self.encoder(self.encoder_input)[2])

        self.model = Model(self.encoder_input, self.outputs, name='vae')
        if self.print_model:
            self.model.summary()
            keras.utils.plot_model(self.model, to_file=f'{self.save_dir}/vae.png', show_shapes=True)

        original_dim = input_shape[0]
        reconstruction_loss = mse(self.encoder_input, self.outputs)
        # reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.model.add_loss(vae_loss)
        self.model.compile(optimizer='adam')

    def create_encoder(self, input_shape) -> (Input, Model, Dense, Dense):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.n_hidden, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        z_mean = Dense(self.n_hidden / 2, name='z_mean')(x)
        z_mean = keras.layers.BatchNormalization()(z_mean)
        z_log_var = Dense(self.n_hidden / 2, name='z_log_var')(x)
        z_log_var = keras.layers.BatchNormalization()(z_log_var)
        z = Lambda(self.sampling, output_shape=(self.n_hidden / 2,), name='z_sampling')([z_mean, z_log_var])
        z = keras.layers.BatchNormalization()(z)
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        if self.print_model:
            encoder.summary()
            keras.utils.plot_model(encoder, to_file=f'{self.save_dir}/vae_encoder.png', show_shapes=True)
        return inputs, encoder, z_mean, z_log_var

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]  # Returns the shape of tensor or variable as a tuple of int or None entries.
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def decoder_model(self, input_shape):
        latent_inputs = Input(shape=(int(self.n_hidden / 2),), name='z_sampling')
        x = Dense(self.n_hidden, activation='relu')(latent_inputs)
        x = keras.layers.BatchNormalization()(x)
        outputs = Dense(input_shape[0], activation='sigmoid')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        if self.print_model:
            decoder.summary()
            keras.utils.plot_model(decoder, to_file=f'{self.save_dir}/vae_decoder.png', show_shapes=True)
        return decoder