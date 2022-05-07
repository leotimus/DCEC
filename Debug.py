import keras.losses

import reader.SequenceReader as sr
from ConvAE2 import CAE2
from DCEC import DCEC
from datasets import load_fasta, get_sequence_samples
import tensorflow as tf
from writer.BinWriter import write_bins, map_bin_by_contig_name


def print_gpu_info():
    contigs = sr.readContigs("/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta")
    print(f'Parsed {len(contigs.keys())} contigs')
    print(f'GPUs Available: {tf.config.list_physical_devices("GPU")}')


def write_bin_samples():
    x = get_sequence_samples()
    dcec = DCEC(filters=[32, 64, 128, 60], n_clusters=60, contig_len=20000)
    dcec.model.load_weights("results/debug2/dcec_model_final.h5")
    clusters = dcec.predict(x, batch_size=256)

    fasta = "/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
    # fastaDict = sr.readContigs(fasta, numberOfSamples=10000, onlySequence=False)
    fastaDict = sr.readContigs(fasta, onlySequence=False)
    binsDict = map_bin_by_contig_name(fastaDict, clusters)
    write_bins("results/debug2/bins", bins=binsDict, fastadict=fastaDict)
    print(f'predict size: ', len(clusters))


def training_full_20k():
    x = get_sequence_samples()
    y = None
    dcec = DCEC(filters=[32, 64, 128, 60], n_clusters=60, contig_len=20000)
    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
    # Step 1: pretrain if necessary
    dcec.init_cae(batch_size=256,
                  cae_weights=None, save_dir='results/debug2', x=x)
    # Step 2: train with cpu
    with tf.device('/cpu:0'):
        dcec.fit(x, y=y, tol=0.001, maxiter=200, update_interval=5, save_dir='results/debug2', batch_size=256)


def dcec_2k_1k():
    x = get_sequence_samples(n_samples=2000)
    y = None
    contig_len = 1000
    optimizer = 'adam'
    with tf.device('/cpu:0'):
        dcec = DCEC(filters=[32, 64, 128, 60], n_clusters=60, contig_len=contig_len)
        dcec.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
        tf.keras.utils.plot_model(dcec.model, to_file='results/tmp/dcec_model.png', show_shapes=True)
        # dcec.compile(loss=loss, optimizer=optimizer)
        dcec.init_cae(batch_size=256, save_dir='results/tmp', x=x)
        dcec.fit(x, y=y, tol=0.001, maxiter=20, update_interval=5, save_dir='results/tmp', batch_size=256)


def verify_cae():
    cae = CAE2(filters=[32, 64, 128, 60], contig_len=10000)
    cae.load_weights("results/temp/pretrain_cae_model.h5")
    x = get_sequence_samples(n_samples=2000)
    from tensorflow import keras
    # feature_model = keras.models(inputs=cae.input, outputs=cae.get_layer(name='embedding').output)
    from reader.DataGenerator import DataGenerator
    cae_generator = DataGenerator(x, batch_size=256, contig_len=10000)
    decodes = cae.predict(x=cae_generator)
    return decodes


if __name__ == "__main__":
    verify_cae()
    print(verify_cae())
