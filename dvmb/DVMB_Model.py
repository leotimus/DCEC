from pathlib import Path
from time import time
import numpy as np
import keras.backend as K
from keras.losses import mse, kld
from tensorflow import keras
from sklearn.cluster import KMeans
from vae.VAE_Model import VAE_Model
import tensorflow as tf


class ClusteringLayer(keras.layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = keras.layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DVMB_Model(keras.models.Model):
    def __init__(self, n_hidden=64, batch_size=256, n_epoch=500, n_clusters=10, save_dir='results/vae2', **kwargs):

        super(DVMB_Model, self).__init__(**kwargs)
        self.save_dir = save_dir
        self.n_clusters = n_clusters
        self.pretrained = False
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.y_pred = []

        self.vae: VAE_Model = VAE_Model(n_hidden=n_hidden, input_shape=(104,), save_dir=save_dir)
        self.vae.compile(optimizer='Adam')

        sampling_layer = self.vae.encoder(self.vae.encoder_input)
        encoded_mean = sampling_layer[0]
        self.clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(encoded_mean)

        # Define DVMB model
        self.build()
        self(inputs=self.vae.encoder_input, outputs=[self.clustering_layer, self.vae.outputs])

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.vae_loss_tracker = keras.metrics.Mean(name="vae_loss")
        self.clustering_tracker = keras.metrics.Mean(name="clustering_loss")

    def pretrain(self, x, batch_size=256, epochs=500):
        print('...Pretraining...')
        # begin training
        t0 = time()
        self.vae.fit(x=x, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: ', time() - t0)
        # self.vae.save(f'{self.save_dir}/pretrain_vae_model.h5')
        # print(f'Pretrained weights are saved to {self.save_dir}/pretrain_vae_model.h5')
        self.pretrained = True

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.vae.encoder.predict(x)

    def predict(self, x):
        q, tmp = self.predict(x=x, batch_size=self.batch_size, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.clustering_tracker,
            self.vae_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            vae_reconstruct = self.decoder(z)

            vae_reconstruct_loss = mse(data, vae_reconstruct)
            vae_reconstruct_loss = tf.reduce_sum(vae_reconstruct_loss)
            vae_reconstruct_loss = tf.reduce_mean(vae_reconstruct_loss)
            vae_reconstruct_loss *= self.inputs_shape
            vae_kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            vae_kl_loss = K.sum(vae_kl_loss, axis=-1)
            vae_kl_loss *= -0.5
            vae_loss = K.mean(vae_reconstruct_loss + vae_kl_loss)

            clustering_output = self.clustering_layer(z_mean)
            clustering_loss = kld(data, clustering_output)

            total_loss = K.mean(vae_loss + clustering_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.vae_loss_tracker.update_state(vae_loss)
        self.clustering_tracker.update_state(clustering_loss)
        losses = {
            "loss": self.total_loss_tracker.result(),
            "clustering_loss": self.clustering_tracker.result(),
            "vae_loss": self.vae_loss_tracker.result()
        }
        return losses

    def fit(self, x, batch_size=256, maxiter=2e3, tol=1e-3, update_interval=140):

        t0 = time()
        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print(f'Initializing cluster centers with k-means {self.n_clusters}.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

        latent_space = self.vae.encoder.predict(x=x)
        mean = latent_space[0]
        self.y_pred = kmeans.fit_predict(mean)
        y_pred_last = np.copy(self.y_pred)
        self.get_layer("clustering").set_weights([kmeans.cluster_centers_])

        # Step 3: deep clust
        save_interval = x.shape[0] / batch_size * 5
        print(f'MaxIter: {maxiter}, Save interval: {save_interval}, Update interval: {update_interval}.')

        index = 0
        size = len(x)
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, tmp = self.model.predict(x=x, batch_size=self.batch_size, verbose=0)
                del tmp
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * batch_size >= size:
                x_ = x[index * batch_size::]
                y_ = p[index * batch_size::]
                loss = self.train_on_batch(x=x_, y=[y_, x_])
                index = 0
            else:
                x_ = x[index * batch_size:(index + 1) * batch_size]
                y_ = p[index * batch_size:(index + 1) * batch_size]
                loss = self.train_on_batch(x=x_, y=[y_, x_])
                index += 1
            print(f'Observe loss: {loss}')
            del x_
            del y_

            # save intermediate model
            if ite != 0 and ite % save_interval == 0:
                # save DVMB model checkpoints
                Path(f'{self.save_dir}/cp').mkdir(parents=True, exist_ok=True)
                file = f'{self.save_dir}/cp/dcec_model_{str(ite)}.h5'
                print(f'saving model to: {file} of iteration={ite}')
                self.save_weights(file)
                print(f'saved model to: {file} of iteration={ite}.')
            ite += 1

        # save the trained model
        # print(f'saving model to: {self.save_dir}/dcec_model_final.h5')
        # self.save_weights(f'{self.save_dir}/dcec_model_final.h5')
        t2 = time()
        print('Clustering time:', t2 - t1)
        print('Total time:     ', t2 - t0)

    def init_vae(self, vae_weights=None, x=None, epoch=500):
        t0 = time()
        if not self.pretrained and vae_weights is None:
            print(f'pretraining VAE using default hyper-parameters:')
            print(f'optimizer=\'adam\', epochs={epoch}')
            self.pretrain(x, self.batch_size, epoch)
            self.pretrained = True
        # elif vae_weights is not None:
            # self.vae.load_weights(vae_weights)
            # print('vae_weights is loaded successfully.')
        print('Pretrain time:  ', time() - t0)
