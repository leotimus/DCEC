import os
from pathlib import Path

import vamb
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.data import Dataset
from vamb.__main__ import calc_tnf, calc_rpkm
from vamb.vambtools import numpy_inplace_maskarray, zscore, write_npz

from vae.VAE import VAE1


def run_vae_mnist():
    global vae
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (original_dim,)
    save_dir = 'results/vae-mnist'
    batch_size, n_epoch = 100, 100
    n_hidden, z_dim = 256, 2
    vae = VAE1(batch_size=batch_size, n_epoch=n_epoch, n_hidden=n_hidden, z_dim=z_dim, input_shape=input_shape)

    vae.vae.fit(x_train, epochs=n_epoch, batch_size=batch_size, validation_data=(x_test, None))
    vae.vae.save_weights(f'vae_mlp_mnist_latent_dim_{z_dim}.h5')

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
    encoder_input = np.expand_dims(x_test[0], axis=0)
    mean, _, _ = vae.encoder.predict(encoder_input)
    decoded_output = vae.decoder.predict(mean)
    plt.ioff()
    img = np.reshape(x_test[0], (image_size, image_size))
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


def run_vae_tnf_bam():
    save_dir, outdir = 'results/vae', 'results/vae'
    batch_size, n_epoch = 100, 500
    n_hidden = 64
    destroy = False

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    bams_path = ['/share_data/cami_low/bams/RL_S001.bam']
    fasta_path = '/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'
    tnf_path = f'{save_dir}/tnf.npz'
    names_path = f'{save_dir}/names.npz'
    lengths_path = f'{save_dir}/lengths.npz'
    rpkm_path = f'{save_dir}/rpkm.npz'
    with open(f'{save_dir}/vectors.log', 'w') as logfile:
        tnf, contignames, contiglengths = calc_tnf(outdir=save_dir,
                                                    fastapath=None,
                                                    tnfpath=tnf_path,
                                                    namespath=names_path,
                                                    lengthspath=lengths_path,
                                                    mincontiglength=100,
                                                    logfile=logfile)
        if not Path(names_path).exists():
            write_npz(os.path.join(save_dir, 'names.npz'), contignames)
        rpkm = calc_rpkm(outdir=save_dir,
                            bampaths=None,
                            rpkmpath=rpkm_path,
                            jgipath=None, mincontiglength=100, refhash=None, ncontigs=len(tnf), minalignscore=None,
                            minid=None,
                            subprocesses=min(os.cpu_count(), 8), logfile=logfile)

    # return run_vamb_ptorch(batch_size, logfile, outdir, rpkm, tnf)

    mask = tnf.sum(axis=1) != 0

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

    if mask.sum() < batch_size:
        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy:
        rpkm = numpy_inplace_maskarray(rpkm, mask)
        tnf = numpy_inplace_maskarray(tnf, mask)
    else:
        # The astype operation does not copy due to "copy=False", but the masking
        # operation does.
        rpkm = rpkm[mask].astype(np.float32, copy=False)
        tnf = tnf[mask].astype(np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        zscore(rpkm, axis=0, inplace=True)

    zscore(tnf, axis=0, inplace=True)

    # TODO improve
    inputs = []
    for idx, x in enumerate(tnf):
        tmp = np.append(rpkm[idx], x)
        inputs.append(tmp)
    inputs = np.array(inputs)
    # TODO improve
    vae = VAE1(batch_size=batch_size, n_epoch=n_epoch,
               n_hidden=n_hidden, input_shape=(104,), print_model=True, save_dir=save_dir)
    vae.vae.fit(x=inputs, epochs=n_epoch, batch_size=batch_size)


def run_vamb_ptorch(batch_size, logfile, outdir, rpkm, tnf):
    nsamples = rpkm.shape[1]
    vae = vamb.encode.VAE(nsamples, nhiddens=None, nlatent=32,
                          alpha=None, beta=200, dropout=None, cuda=False)
    print(vae)
    dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, batch_size,
                                                   destroy=True, cuda=False)
    vamb.vambtools.write_npz(os.path.join(outdir, 'mask.npz'), mask)
    n_discarded = len(mask) - mask.sum()
    print('', file=logfile)
    modelpath = os.path.join(outdir, 'model.pt')
    vae.trainmodel(dataloader, nepochs=500, lrate=1e-3, batchsteps=[25, 75, 150, 300],
                   logfile=logfile, modelfile=modelpath)
    print('', file=logfile)
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del vae  # Needed to free "latent" array's memory references?
    return mask, latent


if __name__ == "__main__":
    run_vae_tnf_bam()
    # run_vae_mnist()