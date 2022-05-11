import reader.SequenceReader as sr
import numpy as np
from DCEC import DCEC
from ConvAE2 import CAE2
from datasets import load_fasta, get_sequence_samples
import tensorflow as tf
from writer.BinWriter import writeBins, mapBinAndContigNames


def print_gpu_info():
    contigs = sr.readContigs("/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta")
    print(f'Parsed {len(contigs.keys())} contigs')
    print(f'GPUs Available: {tf.config.list_physical_devices("GPU")}')


def write_bin_samples():
    x,y = get_sequence_samples()
    dcec = DCEC(filters=[32, 64, 128, 60, 256], n_clusters=60, contig_len=10000)
    dcec.model.load_weights("results/temp/dcec_model_20.h5")
    clusters = dcec.predict(x, batch_size=256)

    fasta = "/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
    # fastaDict = sr.readContigs(fasta, numberOfSamples=10000, onlySequence=False)
    fastaDict = sr.readContigs(fasta, onlySequence=False)
    binsDict = mapBinAndContigNames(fastaDict, clusters)
    writeBins("results/debug1/bins", bins=binsDict, fastadict=fastaDict)
    print(f'predict size: ', len(clusters))



def training_full_20k():
    #x = get_sequence_samples()
    #y = None
    x, y = get_sequence_samples()
    y = np.array(y).astype(np.int64)
    dcec = DCEC(filters=[32, 64, 128, 60, 256], n_clusters=53, contig_len=10000)
    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', tf.keras.losses.CosineSimilarity()], loss_weights=[0.1, 1], optimizer=optimizer)
    # Step 1: pretrain if necessary
    dcec.init_cae(batch_size=256,
                  cae_weights='results/cosineSimilarity/pretrain_cae_model.h5',
                  save_dir='results/temp', x=x)
    # Step 2: train with cpu
    with tf.device('/cpu:0'):
        dcec.fit(x, y=y, tol=0.001, maxiter=200, update_interval=140, save_dir='results/debug2', batch_size=256)

def verify_cae():
    cae = CAE2(filters=[32, 64, 128, 60, 256], contig_len=10000)
    cae.load_weights("results/temp/pretrain_cae_model.h5")
    x,y = get_sequence_samples()
    from tensorflow import keras
    # feature_model = keras.models(inputs=cae.input, outputs=cae.get_layer(name='embedding').output)
    from reader.DataGenerator import DataGenerator
    cae_generator = DataGenerator(x, batch_size=256, contig_len=10000)
    decodes = cae.predict(x=cae_generator)
    return decodes

if __name__ == "__main__":
    #write_bin_samples()
    decodes = training_full_20k()
    #print(decodes)
