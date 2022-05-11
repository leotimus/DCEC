import keras.losses

import reader.SequenceReader as sr
from ConvAE2 import CAE2
from DCEC import DCEC
from datasets import load_fasta, get_sequence_samples, load_mnist, load_mnist2
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
        tf.keras.utils.plot_model(dcec.model, to_file='results/tmp3/dcec_model.png', show_shapes=True)
        # dcec.compile(loss=loss, optimizer=optimizer)
        dcec.init_cae(batch_size=256, save_dir='results/tmp3', x=x)
        dcec.fit(x, y=y, tol=0.001, maxiter=20, update_interval=5, save_dir='results/tmp3', batch_size=256)


def verify_cae():
    cae = CAE2(filters=[32, 64, 128, 60], contig_len=20000)
    cae.load_weights("results/debug2/pretrain_cae_model.h5")
    from tensorflow import keras
    # feature_model = keras.models(inputs=cae.input, outputs=cae.get_layer(name='embedding').output)
    # x = get_sequence_samples(n_samples=2000)
    # from reader.DataGenerator import DataGenerator
    # cae_generator = DataGenerator(x, batch_size=256, contig_len=20000)
    # decodes = cae.predict(x=cae_generator)

    x, y = load_fasta(n_samples=2000, contig_len=20000)
    decodes = cae.predict(x=x)
    return decodes


def verify_ae():
    from AE import create_auto_encoder_models
    # ae = create_auto_encoder_models(dims=[20000, 2000, 1000, 500, 100], act='relu', init='glorot_uniform')
    ae = create_auto_encoder_models()
    # ae.compile(optimizer='adam', loss='mse')
    ae.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity())
    ae.summary()

    x, y = load_fasta(n_samples=2000, contig_len=1000)
    # from reader.DataGenerator import DataGenerator
    # cae_generator = DataGenerator(x, batch_size=256, contig_len=20000)
    ae.fit(x, x, epochs=10, batch_size=256)
    ae.save('results/ae/pretrain_cae_model.h5')
    reconstruct = ae.predict(x)
    print(f'reconstruct size {reconstruct.shape}')




if __name__ == "__main__":
    # verify_ae()
    # x, y = load_mnist()
    # a, b = load_mnist2()
    # dcec_2k_1k()
    verify_cae()