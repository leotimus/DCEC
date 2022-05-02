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
    dcec = DCEC(filters=[32, 64, 128, 37, 256], n_clusters=37, contig_len=1008)
    dcec.model.load_weights(None)
    clusters = dcec.predict(x, batch_size=256)

    fasta = "/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
    # fastaDict = sr.readContigs(fasta, numberOfSamples=10000, onlySequence=False)
    fastaDict = sr.readContigs(fasta, onlySequence=False)
    binsDict = mapBinAndContigNames(fastaDict, clusters)
    writeBins("results/debug6/bins", bins=binsDict, fastadict=fastaDict)
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
                  cae_weights=None,
                  save_dir='results/debug6/', x=x)
    # Step 2: train with cpu
    with tf.device('/cpu:0'):
        dcec.fit(x, y=y, tol=0.001, maxiter=200, update_interval=5, save_dir='results/debug6/', batch_size=256)


if __name__ == "__main__":
    training_full_20k()
