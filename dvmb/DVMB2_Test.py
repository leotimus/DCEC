import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import sklearn

from dvmb.DVMB2 import DVMB2
from reader.SequenceReader import readContigs
from vae.VAE_Test import get_input
from writer.BinWriter import mapBinAndContigNames, writeBins


def run_deep_clustering(save_dir='results/dvmb_model2',
                        n_clusters=60,
                        batch_size=256,
                        n_epoch=300,
                        n_hidden=64, use_batch_norm=False, input_shape=(104, 0), vae_optimizer='Adam', optimizer='Adam',
                        loss_weights=[0.1, 1], update_interval=100, max_iter=2e3):
    shutil.rmtree(f'{save_dir}/bins', ignore_errors=True, onerror=None)
    destroy = False

    tnf, rpkm = get_input(batch_size, destroy, save_dir='/share_data/cami_low/npzs')
    tnf = sklearn.preprocessing.minmax_scale(tnf, feature_range=(0, 1), axis=1, copy=True)
    rpkm = sklearn.preprocessing.minmax_scale(rpkm, feature_range=(0, 1), axis=0, copy=True)

    print(f'x_max={np.max(tnf)}, x_min={np.min(tnf)}')
    print(f'x1_max={np.max(rpkm)}, x1_min={np.min(rpkm)}')

    inputs = []
    for idx, tnf in enumerate(tnf):
        tmp = np.append(rpkm[idx], tnf)
        inputs.append(tmp)
    tnf = np.array(inputs)

    # dvmb: DVMB2 = DVMB2(n_hidden=n_hidden, batch_size=batch_size, n_epoch=n_epoch, n_clusters=n_clusters,
    #                               save_dir=save_dir)
    dvmb: DVMB2 = DVMB2(n_hidden=n_hidden, batch_size=batch_size, n_epoch=n_epoch, n_clusters=n_clusters,
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

    n_clusters = 60
    batch_size = 256
    n_epoch = 300
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

    case = 'default_normal_with_plot'
    # max_iters = range(1, 5000, 5)
    max_iters = [1000]

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
                                    max_iter=max_iter)

        subprocess.run(['/home/ltms/Projects/DCEC/amber_runner.sh', save_dir])
