import os
from pathlib import Path

import numpy as np
import sklearn.preprocessing
from keras.datasets import mnist
from keras.optimizer_v2.adam import Adam

from plotting.PlotCallback import PlotCallback
from vae.VAE import VAE
from vae.VAE_Test import get_input
import matplotlib.pyplot as plt


def pretrain_vae_model_mnist():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (original_dim,)
    save_dir = 'results/vae-model-mnist'

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    test = os.listdir(save_dir)
    for item in test:
        if item.endswith(".png"):
            os.remove(os.path.join(save_dir, item))

    batch_size, n_epoch = 100, 100
    n_hidden = 256

    vae: VAE = VAE(input_shape=input_shape, n_hidden=n_hidden, save_dir=save_dir)
    vae.compile(optimizer='adam')
    print(f'x_max={np.max(x_train)}, x_min={np.min(x_train)}')
    plot_callback = PlotCallback(f'{save_dir}/vae-model-mnist-loss0.png', ['loss', 'reconstruction_loss'])
    plot1_callback = PlotCallback(f'{save_dir}/vae-model-mnist-loss1.png', ['kl_loss'])
    vae.fit(x_train, epochs=n_epoch, batch_size=batch_size, callbacks=[plot_callback, plot1_callback])
    vae.save_weights(f'{save_dir}/vae_model_final.h5')
    plot_callback.plot()
    plot1_callback.plot()
    # vae.built = True
    # vae.load_weights(f'{save_dir}/vae_model_final.h5')

    z_mean, _, _ = vae.encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(f'{save_dir}/vae_mean.png')

    plt.ioff()
    img = np.reshape(x_test[0], (image_size, image_size))
    plt.axis('off')
    plt.imshow(img, cmap='gray_r')
    plt.savefig(f'{save_dir}/seven_original.png')

    plt.ioff()
    encoder_input = np.expand_dims(x_test[0], axis=0)
    mean, _, _ = vae.encoder.predict(encoder_input)
    decoded_output = vae.decoder.predict(mean)
    plt.axis('off')
    plt.imshow(np.reshape(decoded_output, (image_size, image_size)), cmap='gray_r')
    plt.savefig(f'{save_dir}/seven_reproduced.png')

    plt.ioff()
    plt.figure(figsize=(5, 20))
    for i in range(30):
        plt.subplot(30, 2, 2 * i + 1)
        img = np.reshape(x_test[i], (image_size, image_size))
        plt.axis('off')
        plt.imshow(img, cmap='gray_r')

        plt.subplot(30, 2, 2 * i + 2)
        encoder_input = np.expand_dims(x_test[i], axis=0)
        mean, _, _ = vae.encoder.predict(encoder_input)
        decoded_output = vae.decoder.predict(mean)
        plt.axis('off')
        plt.imshow(np.reshape(decoded_output, (image_size, image_size)), cmap='gray_r')
    plt.savefig(f'{save_dir}/mnist_reproduction.png')


def pretrain_vae_model():
    save_dir = 'results/vae_model-cami'
    batch_size, n_epoch = 256, 300
    n_hidden = 64
    destroy = False
    tnf, rpkm = get_input(batch_size, destroy, save_dir)
    x = sklearn.preprocessing.minmax_scale(tnf, feature_range=(0, 1), axis=1, copy=True)
    x1 = sklearn.preprocessing.minmax_scale(rpkm, feature_range=(0, 1), axis=0, copy=True)

    print(f'x_max={np.max(x)}, x_min={np.min(x)}')
    print(f'x1_max={np.max(x1)}, x1_min={np.min(x1)}')

    inputs = []
    for idx, x in enumerate(x):
        tmp = np.append(x1[idx], x)
        inputs.append(tmp)
    inputs = np.array(inputs)

    input_shape = (104,)

    vae: VAE = VAE(input_shape=input_shape, n_hidden=n_hidden, save_dir=save_dir)
    vae.compile(optimizer='Adam')

    plot_callback = PlotCallback(f'/share_data/reports/dvmb/vae-model-cami-loss0.png', ['loss', 'reconstruction_loss'])
    plot1_callback = PlotCallback(f'/share_data/reports/dvmb/vae-model-cami-loss1.png', ['kl_loss'])
    vae.fit(inputs, epochs=n_epoch, batch_size=batch_size, callbacks=[plot_callback, plot1_callback])
    plot_callback.plot()
    plot1_callback.plot()

if __name__ == "__main__":
    pretrain_vae_model()
    # pretrain_vae_model_mnist()