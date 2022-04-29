import keras.losses

import reader.SequenceReader as sr
from DCEC import DCEC
from datasets import load_fasta, get_sequence_samples
import tensorflow as tf
from writer.BinWriter import writeBins, mapBinAndContigNames


def print_gpu_info():
    contigs = sr.readContigs("/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta")
    print(f'Parsed {len(contigs.keys())} contigs')
    print(f'GPUs Available: {tf.config.list_physical_devices("GPU")}')


def write_bin_samples():
    x = get_sequence_samples()
    dcec = DCEC(filters=[32, 64, 128, 10], n_clusters=60, contig_len=20000)
    dcec.model.load_weights("results/debug1/dcec_model_final.h5")
    clusters = dcec.predict(x, batch_size=256)

    fasta = "/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
    # fastaDict = sr.readContigs(fasta, numberOfSamples=10000, onlySequence=False)
    fastaDict = sr.readContigs(fasta, onlySequence=False)
    binsDict = mapBinAndContigNames(fastaDict, clusters)
    writeBins("results/debug1/bins", bins=binsDict, fastadict=fastaDict)
    print(f'predict size: ', len(clusters))


def training_full_20k():
    x = get_sequence_samples()
    y = None
    dcec = DCEC(filters=[32, 64, 128, 10], n_clusters=60, contig_len=20000)
    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
    # Step 1: pretrain if necessary
    dcec.init_cae(batch_size=256,
                  cae_weights='results/temp1/fasta-pretrain-model-10.h5',
                  save_dir='results/debug1/', x=x)
    # Step 2: train with cpu
    with tf.device('/cpu:0'):
        dcec.fit(x, y=y, tol=0.001, maxiter=200, update_interval=5, save_dir='results/debug2', batch_size=256)


def loss(y_true, y_pred):
    mse = keras.losses.MSE(y_true, y_pred)
    kld = keras.losses.KLD(y_true, y_pred)
    # mse_out = tf.reduce_sum(mse(y_true, y_pred))
    # kld_out = tf.reduce_sum(kld(y_true, y_pred))
    print(f'get mse {mse}, kld {kld}, returning the sum {kld * 0.1 + mse * 1}')
    # return kld_out * 0.1 + mse_out * 1, kld_out, mse
    return mse


def dcec_2k_1k():
    x = get_sequence_samples(n_samples=2000)
    y = None
    contig_len = 1000
    optimizer = 'adam'
    with tf.device('/cpu:0'):
        dcec = DCEC(filters=[32, 64, 128, 60], n_clusters=60, contig_len=contig_len)
        # dcec.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
        dcec.compile(loss=loss, optimizer=optimizer)
        dcec.init_cae(batch_size=256, save_dir='results/tmp', x=x)
        dcec.fit(x, y=y, tol=0.001, maxiter=20, update_interval=5, save_dir='results/tmp', batch_size=256)


if __name__ == "__main__":
    dcec_2k_1k()
