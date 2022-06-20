from pathlib import Path
from time import time
import numpy as np
import keras.backend as K
from tensorflow import keras
from sklearn.cluster import KMeans
import metrics
from SAE import SAE


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
        # self.input_spec = keras.layers.InputSpec(ndim=2)
        self.input_spec = keras.layers.InputSpec()

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


class DSMB(object):
    def __init__(self, n_hidden=64, batch_size=256, n_epoch=100, n_clusters=60, save_dir='results/dsmb'):

        super(DSMB, self).__init__()
        self.save_dir = save_dir
        self.n_clusters = n_clusters
        self.pretrained = False
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.y_pred = []
        self.z_log_var = None
        self.z_mean = None
      

        self.sae = SAE(batch_size=batch_size, n_epoch=n_epoch,
                       n_hidden=n_hidden, input_shape=(104, ), print_model=True, save_dir=save_dir)


        encoder_latent_space_layer = self.sae.encoder(self.sae.encoder_input)
        self.clustering_layer = ClusteringLayer(60, name='clustering')(encoder_latent_space_layer)

      

        self.model = keras.models.Model(inputs=self.sae.encoder_input,
                                        outputs=[self.clustering_layer, self.sae.outputs])
        keras.utils.plot_model(self.model, to_file=f'{save_dir}/dsmb.png', show_shapes=True)

    def pretrain(self, x, batch_size=256, epochs=100, optimizer='adam'):
        print('...Pretraining...')
        self.sae.model.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(f'{self.save_dir}/pretrain_log.csv')

        # begin training
        t0 = time()
        self.sae.model.fit(x=x, y=x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print('Pretraining time: ', time() - t0)
        self.sae.model.save(f'{self.save_dir}/pretrain_sae_model.h5')
        print(f'Pretrained weights are saved to {self.save_dir}/pretrain_sae_model.h5, reload weights')
        self.sae.model.load_weights(f'{self.save_dir}/pretrain_sae_model.h5')
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.sae.encoder.predict(x)

    def predict(self, x, batch_size):

        # q, _ = self.model.predict(x, verbose=0)
        # return q.argmax(1)

        q = None
        complete = False
        p_index = 0
        size = len(x)
        while not complete:
            x_ = x[p_index * batch_size:(p_index + 1) * batch_size]
            q_, tmp = self.model.predict(x=x_, batch_size=None, verbose=0)
            del tmp
            if q is None:
                q = q_
            else:
                q = np.append(q, q_, axis=0)
            if (p_index + 1) * batch_size >= size:
                complete = True
                del q_
                del x_
            else:
                p_index += 1
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


    def compile(self, loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
    
    def fit(self, x, y=None, batch_size=256, maxiter=20000, tol=1e-2, update_interval=150):
        t0 = time()
        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print(f'Initializing cluster centers with k-means {self.n_clusters}.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        latent_space = self.sae.encoder.predict(x=x)
        z = latent_space
        self.y_pred = kmeans.fit_predict(z)
        y_pred_last = np.copy(self.y_pred)

        self.model.get_layer("clustering").set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logfile = open(f'{self.save_dir}/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        # save_interval = x.shape[0] / batch_size * 5
        # save_interval = len(self.y_pred) / batch_size * 5
        save_interval = 150
        print(f'MaxIter: {maxiter}, Save interval: {save_interval}, Update interval: {update_interval}.')

        loss = [0, 0, 0]
        index = 0
        size = len(x)
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:

                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)
                self.y_pred = q.argmax(1)
                # predict on batch
                # q = None
                # complete = False
                # p_index = 0
                # while not complete:
                #     x_ = x[p_index * batch_size:(p_index + 1) * batch_size]
                #     q_, tmp = self.model.predict(x=x_, batch_size=None, verbose=0)
                #     del tmp
                #     if q is None:
                #         q = q_
                #     else:
                #         q = np.append(q, q_, axis=0)
                #     if (p_index + 1) * batch_size >= size:
                #         complete = True
                #         del q_
                #         del x_
                #     else:
                #         p_index += 1
                # p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                print(delta_label)
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break


                loss = self.model.train_on_batch(x=x, y=[p, x])
                print(f'Observe loss: {loss}.')

            # save intermediate model
            if ite != 0 and ite % save_interval == 0:
                # save DSMB model checkpoints
                Path(f'{self.save_dir}/cp').mkdir(parents=True, exist_ok=True)
                file = f'{self.save_dir}/cp/dcec_model_{str(ite)}.h5'
                print(f'saving model to: {file} of iteration={ite}')
                self.model.save_weights(file)
                print(f'saved model to: {file} of iteration={ite}.')

            ite += 1

        # save the trained model
        logfile.close()
        print(f'saving model to: {self.save_dir}/dcec_model_final.h5')
        self.model.save_weights(f'{self.save_dir}/dcec_model_final.h5')
        t2 = time()
        print('Clustering time:', t2 - t1)
        print('Total time:     ', t2 - t0)

    def init_sae(self, sae_weights=None, x=None):
        t0 = time()

        print(f'pretraining SAE using default hyper-parameters:')
        print(f'optimizer=\'adam\', epochs={self.n_epoch}')
        self.pretrain(x, self.batch_size)
        self.pretrained = True