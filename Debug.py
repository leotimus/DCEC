import os

from vamb.__main__ import calc_tnf, calc_rpkm

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


def test_tnf_abd(logfile):
    tnfs, contignames, contiglengths = calc_tnf(outdir='results/vectors',
                                                fastapath='/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta',
                                                tnfpath=None, namespath=None, lengthspath=None, mincontiglength=100,
                                                logfile=logfile)
    rpkms = calc_rpkm(outdir='results/vectors', bampaths=['/share_data/cami_low/bams/RL_S001.bam'], rpkmpath=None,
                      jgipath=None, mincontiglength=100, refhash=None, ncontigs=len(tnfs), minalignscore=None, minid=None,
                      subprocesses=min(os.cpu_count(), 8), logfile=logfile)
    print(f'tnfs shape: {tnfs.shape}, rpkms shape: {rpkms}')


if __name__ == "__main__":
    with open('results/vectors.log', 'w') as logfile:
        test_tnf_abd(logfile)
