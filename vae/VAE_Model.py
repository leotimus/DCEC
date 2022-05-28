import keras.backend as K
import tensorflow as tf
from keras import Input
from keras.layers import Dense, BatchNormalization, Layer
from keras.losses import mse
from keras.metrics import Mean
from keras.models import Model
from tensorflow import keras


class VAE_Model(Model):
    def __init__(self, input_shape=None, n_hidden=128, save_dir='results/tmp', **kwargs):
        super(VAE_Model, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.print_model = True
        self.save_dir = save_dir
        self.inputs_shape = input_shape

        self.encoder_input, self.encoder, self.z_mean, self.z_log_var = self.create_encoder(input_shape)
        self.decoder = self.decoder_model(input_shape)
        self_encoder_outputs = self.encoder(self.encoder_input)
        mean_output = self_encoder_outputs[2]
        self.decoder_outputs = self.decoder(mean_output)

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            kl_loss, reconstruction_loss, total_loss = self.calculate_vae_loss(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        losses = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        return losses

    def calculate_vae_loss(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = mse(data, reconstruction)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        reconstruction_loss *= self.inputs_shape
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return kl_loss, reconstruction_loss, total_loss

    def create_encoder(self, input_shape) -> (Input, Model, Dense, Dense):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.n_hidden, activation='relu')(inputs)
        # x = BatchNormalization()(x)
        z_mean = Dense(self.n_hidden / 2, name="z_mean")(x)
        # z_mean = BatchNormalization()(z_mean)
        z_variant = Dense(self.n_hidden / 2, name="z_log_var")(x)
        # z_variant = BatchNormalization()(z_variant)
        z = Sampling()([z_mean, z_variant])
        # z = BatchNormalization()(z)
        encoder = Model(inputs, [z_mean, z_variant, z], name='encoder')
        if self.print_model:
            encoder.summary()
            keras.utils.plot_model(encoder, to_file=f'{self.save_dir}/vae_encoder.png', show_shapes=True)
        return inputs, encoder, z_mean, z_variant

    def decoder_model(self, input_shape) -> Model:
        latent_inputs = Input(shape=(int(self.n_hidden / 2),), name='z_sampling')
        x = Dense(self.n_hidden, activation='relu')(latent_inputs)
        # x = BatchNormalization()(x)
        outputs = Dense(input_shape[0], activation='sigmoid')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        if self.print_model:
            decoder.summary()
            keras.utils.plot_model(decoder, to_file=f'{self.save_dir}/vae_decoder.png', show_shapes=True)
        return decoder


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon