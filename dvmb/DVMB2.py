from time import time
import numpy as np
import keras.backend as K
from keras.losses import mse, kld
from tensorflow import keras
from sklearn.cluster import KMeans
from plotting.PlotCallback import PlotCallback
from vae.VAE_Model import VAE_Model


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
        2D tensor with shape: `(n_samples, n_clusters)`.s
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


class DVMB2(object):
    def __init__(self, n_hidden=64, batch_size=256, n_epoch=500, n_clusters=10, save_dir='results/vae2',
                 use_batch_norm=False, input_shape=(104,), vae_optimizer='Adam', optimizer='Adam',
                 loss_weights=[0.1, 1], update_interval=100, max_iter=2e3):

        super(DVMB2, self).__init__()
        self.tol = 0.01
        self.update_interval = update_interval
        self.max_iter = max_iter
        self.save_dir = save_dir
        self.n_clusters = n_clusters
        self.pretrained = False
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.y_pred = []
        self.use_batch_norm = use_batch_norm
        self.optimizer = optimizer
        self.loss_weights = loss_weights

        self.vae = VAE_Model(n_hidden=n_hidden, input_shape=input_shape, save_dir=save_dir, use_batch_norm=use_batch_norm)
        self.vae.compile(optimizer=vae_optimizer)

        sampling_layer = self.vae.encoder(self.vae.encoder_input)
        encoder_latent_space_layer = sampling_layer[0]
        self.clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(encoder_latent_space_layer)

        # Define DVMB model
        self.model: keras.models.Model = keras.models.Model(inputs=self.vae.encoder_input,
                                                            outputs=[self.clustering_layer, self.vae.decoder_outputs])

        self.model.summary()
        keras.utils.plot_model(self.model, to_file=f'{save_dir}/dvmb.png', show_shapes=True)

        self.dvmb_loss = PlotCallback(filename=f'{save_dir}/dvmb_loss_plot.png',
                                                  loss_metrics=['loss'])
        self.dvmb_decoderloss = PlotCallback(filename=f'{save_dir}/dvmb_decoderloss_plot.png',
                                                  loss_metrics=['decoder_loss'])
        self.dvmb_clusteringloss = PlotCallback(filename=f'{save_dir}/dvmb_clusteringloss_plot.png',
                                                  loss_metrics=['clustering_loss'])

    def pretrain(self, x):
        print('...Pretraining...')
        t0 = time()
        loss_reLoss_callback = PlotCallback(filename=f'{self.save_dir}/vae_loss_recon_plot.png', loss_metrics=['loss', 'reconstruction_loss'])
        kld_callback = PlotCallback(filename=f'{self.save_dir}/vae_kld_plot.png', loss_metrics=['kl_loss'])
        self.vae.fit(x=x, batch_size=self.batch_size, epochs=self.n_epoch, callbacks=[loss_reLoss_callback, kld_callback])
        loss_reLoss_callback.plot()
        kld_callback.plot()
        print('Pretraining time: ', time() - t0)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.vae.encoder.predict(x)

    def predict(self, x):
        q, tmp = self.model.predict(x=x, batch_size=self.batch_size, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def custom_vae_loss(self, vae):

        def loss(y_true, y_pred):
            _, _, total_loss = vae.calculate_vae_loss(y_true)
            return total_loss

        return loss

    def custom_kld_loss(self):

        def loss(y_true, y_pred):
            return kld(y_true, y_pred)

        return loss

    def compile(self):
        self.model.compile(loss=[self.custom_kld_loss(), self.custom_vae_loss(self.vae)], loss_weights=self.loss_weights,
                           optimizer=self.optimizer)

    def fit(self, x):
        t0 = time()
        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print(f'Initializing cluster centers with k-means {self.n_clusters}.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        latent_space = self.vae.encoder.predict(x=x)
        z = latent_space[0]
        self.y_pred = kmeans.fit_predict(z)
        y_pred_last = np.copy(self.y_pred)

        self.model.get_layer("clustering").set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        save_interval = x.shape[0] / self.batch_size * 5
        print(f'MaxIter: {self.max_iter}, Save interval: {save_interval}, Update interval: {self.update_interval}.')

        index = 0
        ds_size = len(x)

        for ite in range(int(self.max_iter)):
            if ite % self.update_interval == 0:
                q, tmp = self.model.predict(x=x, batch_size=self.batch_size, verbose=0)
                del tmp

                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # check stop criterion
                self.y_pred = q.argmax(1)
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < self.tol:
                    print('delta_label ', delta_label, '< tol ', self.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            loss = []
            if (index + 1) * self.batch_size >= ds_size:
                x_ = x[index * self.batch_size::]
                y_ = p[index * self.batch_size::]
                loss = self.model.train_on_batch(x=x_, y=[y_, x_])
                index = 0
            else:
                x_ = x[index * self.batch_size:(index + 1) * self.batch_size]
                y_ = p[index * self.batch_size:(index + 1) * self.batch_size]
                loss = self.model.train_on_batch(x=x_, y=[y_, x_])
                index += 1
            del x_
            del y_

            self.dvmb_loss.on_epoch_end(ite, logs={'loss': loss[0]})
            self.dvmb_decoderloss.on_epoch_end(ite, logs={'decoder_loss': loss[2]})
            self.dvmb_clusteringloss.on_epoch_end(ite, logs={'clustering_loss': loss[1]})

            ite += 1

        t2 = time()
        print('Clustering time:', t2 - t1)
        print('Total time:     ', t2 - t0)
        self.dvmb_clusteringloss.plot()
        self.dvmb_loss.plot()
        self.dvmb_decoderloss.plot()

    def init_vae(self, x):
        t0 = time()
        print(f'pretraining VAE using default hyper-parameters:')
        print(f'optimizer={self.optimizer}, epochs={self.n_epoch}')
        self.pretrain(x)
        print('Pretrain time:  ', time() - t0)
