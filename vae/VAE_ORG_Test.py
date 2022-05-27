import os
from pathlib import Path

import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

from vae.VAE_ORG import VAE_ORG


def run_vae_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (original_dim,)

    save_dir = 'results/vae-org-mnist'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    test = os.listdir(save_dir)
    for item in test:
        if item.endswith(".png"):
            os.remove(os.path.join(save_dir, item))


    batch_size, n_epoch = 100, 100
    n_hidden, z_dim = 256, 2

    vae = VAE_ORG(batch_size=batch_size, n_epoch=n_epoch, n_hidden=n_hidden, input_shape=input_shape, z_dim=z_dim)
    vae.vae.fit(x_train, epochs=n_epoch, batch_size=batch_size, validation_data=(x_test, None))
    vae.vae.save_weights(f'{save_dir}/vae_mlp_mnist_latent_dim_{z_dim}.h5')
    # vae.vae.load_weights(f'{save_dir}/vae_mlp_mnist_latent_dim_{z_dim}.h5')

    # vae.vae.load_weights(f'{save_dir}/vae.h5')
    filename = f'{save_dir}/vae_mean.png'
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(filename)
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
        img = np.reshape(x_test[0], (image_size, image_size))
        plt.axis('off')
        plt.imshow(np.reshape(decoded_output, (image_size, image_size)), cmap='gray_r')
    plt.savefig(f'{save_dir}/mnist_reproduction.png')

    plt.ioff()
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='gray_r')
    plt.savefig(f'{save_dir}/all.png')

if __name__ == "__main__":
    run_vae_mnist()