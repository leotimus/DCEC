import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import sklearn
from keras.utils.tf_utils import set_random_seed
from vamb.__main__ import calc_tnf, calc_rpkm
from vamb.vambtools import numpy_inplace_maskarray, write_npz

from dvmb.DVMB import DVMB
from reader.SequenceReader import readContigs
from writer.BinWriter import mapBinAndContigNames, writeBins
import tensorflow as tf

#disable random
set_random_seed(2)
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def get_input(batch_size, destroy, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    bams_path = ['/share_data/cami_low/bams/RL_S001.bam']
    fasta_path = '/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'

    tnf_path = f'{save_dir}/tnf.npz'
    if not Path(tnf_path).exists():
        tnf_path = None
    else:
        fasta_path = None

    names_path = f'{save_dir}/names.npz'

    lengths_path = f'{save_dir}/lengths.npz'
    if not Path(lengths_path).exists():
        lengths_path = None

    rpkm_path = f'{save_dir}/rpkm.npz'
    if not Path(rpkm_path).exists():
        rpkm_path = None
    else:
        bams_path = None

    with open(f'{save_dir}/vectors.log', 'w') as logfile:
        tnf, contignames, contiglengths = calc_tnf(outdir=save_dir,
                                                   fastapath=fasta_path,
                                                   tnfpath=tnf_path,
                                                   namespath=names_path,
                                                   lengthspath=lengths_path,
                                                   mincontiglength=100,
                                                   logfile=logfile)
        if not Path(names_path).exists():
            write_npz(names_path, contignames)
        rpkm = calc_rpkm(outdir=save_dir,
                         bampaths=bams_path,
                         rpkmpath=rpkm_path,
                         jgipath=None, mincontiglength=100, refhash=None, ncontigs=len(tnf), minalignscore=None,
                         minid=None,
                         subprocesses=min(os.cpu_count(), 8), logfile=logfile)
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
    """
        if rpkm.shape[1] > 1:
            rpkm /= depthssum.reshape((-1, 1))
        else:
            zscore(rpkm, axis=0, inplace=True)

        zscore(tnf, axis=0, inplace=True)
        """
    # TODO improve
    # inputs = []
    # for idx, x in enumerate(tnf):
    #     tmp = np.append(rpkm[idx], x)
    #     inputs.append(tmp)
    # inputs = np.array(inputs)
    return tnf, rpkm


def run_deep_clustering(save_dir='results/dvmb_model2',
                        n_clusters=60,
                        batch_size=256,
                        n_epoch=300,
                        n_hidden=64, use_batch_norm=False, input_shape=(104, 0), vae_optimizer='Adam', optimizer='Adam',
                        loss_weights=[0.1, 1], update_interval=100, max_iter=2e3, npzs_dir=None):
    shutil.rmtree(f'{save_dir}/bins', ignore_errors=True, onerror=None)
    destroy = False

    if npzs_dir is None:
        npzs_dir = save_dir

    tnf, rpkm = get_input(batch_size, destroy, save_dir=npzs_dir)
    tnf = sklearn.preprocessing.minmax_scale(tnf, feature_range=(0, 1), axis=1, copy=True)
    rpkm = sklearn.preprocessing.minmax_scale(rpkm, feature_range=(0, 1), axis=0, copy=True)

    # print(f'x_max={np.max(tnf)}, x_min={np.min(tnf)}')
    # print(f'x1_max={np.max(rpkm)}, x1_min={np.min(rpkm)}')

    inputs = []
    for idx, tnf in enumerate(tnf):
        tmp = np.append(rpkm[idx], tnf)
        inputs.append(tmp)
    tnf = np.array(inputs)

    # dvmb: DVMB = DVMB(n_hidden=n_hidden, batch_size=batch_size, n_epoch=n_epoch, n_clusters=n_clusters,
    #                               save_dir=save_dir)
    dvmb: DVMB = DVMB(n_hidden=n_hidden, batch_size=batch_size, n_epoch=n_epoch, n_clusters=n_clusters,
                      save_dir=save_dir, use_batch_norm=use_batch_norm, input_shape=input_shape,
                      vae_optimizer=vae_optimizer, optimizer=optimizer, loss_weights=loss_weights,
                      update_interval=update_interval, max_iter=max_iter)
    dvmb.compile()

    # pre-training
    dvmb.init_vae(x=tnf)
    # real training
    # losses = dvmb.vae.calculate_vae_loss(tnf)
    # print(f'evaluate01={evaluate_1}')
    dvmb.fit(x=tnf)
    # losses = dvmb.vae.calculate_vae_loss(tnf)
    # print(f'evaluate02={evaluate_2}')

    # predict
    print(f'DVMB loss labels {dvmb.model.metrics_names}')
    clusters = dvmb.predict(x=tnf)

    # save to bins
    fasta = '/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'
    fastaDict = readContigs(fasta, onlySequence=False)
    binsDict = mapBinAndContigNames(fastaDict, clusters)
    Path(f'{save_dir}/bins').mkdir(exist_ok=True)
    writeBins(f'{save_dir}/bins', bins=binsDict, fastadict=fastaDict)
    print(f'predict size: ', len(clusters))


if __name__ == "__main__":
    root = '/share_data/reports/dvmb'

    n_clusters = 37
    batch_size = 256
    n_epoch = 500
    n_hidden = 64
    use_batch_norm = False
    input_shape = (104,)
    vae_optimizer = 'Adam'
    optimizer = 'Adam'
    loss_weights = [0.1, 1]
    update_interval = 100
    max_iter = 2e3

    # case = 'default_epoch500-1000'
    # lws=[[0.005, 1], [0.01, 1], [0.05, 1], [0.1, 1], [0.1, 0.1], [0.1, 0.5]]

    # case = 'default_epoch500-1000'
    # n_epochs = [1,5,10,15,20,25,30,35,40,45,50]
    # n_epochs = [500,600,700,800,900,1000]

    case = 'test'
    # max_iters = range(1, 5000, 5)
    # max_iters = [1, 50, 100, 400, 900, 1200]
    max_iters = [50]

    for max_iter in max_iters:
        lwss = f'lw{loss_weights[0]}-{loss_weights[1]}'
        save_dir = f'{root}/{case}/n{n_clusters}-b{batch_size}-e{n_epoch}-h{n_hidden}-vo{vae_optimizer}-o{optimizer}-ui{update_interval}-mi{max_iter}-{lwss}'
        print(f'save to {save_dir}')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        file_path = f'{save_dir}/run.txt'
        with open(file_path, "w") as o:
            with contextlib.redirect_stdout(o):
                run_deep_clustering(save_dir=save_dir,
                                    n_clusters=n_clusters,
                                    batch_size=batch_size,
                                    n_epoch=n_epoch,
                                    n_hidden=n_hidden,
                                    use_batch_norm=False,
                                    input_shape=input_shape,
                                    vae_optimizer=vae_optimizer,
                                    optimizer=optimizer,
                                    loss_weights=loss_weights,
                                    update_interval=update_interval,
                                    max_iter=max_iter,
                                    npzs_dir='/share_data/cami_low/npzs')

        subprocess.run(['/home/ltms/Projects/DCEC/amber_runner.sh', save_dir])
