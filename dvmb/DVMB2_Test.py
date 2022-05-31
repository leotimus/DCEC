from pathlib import Path

import numpy as np
import sklearn

from dvmb.DVMB2 import DVMB2
from reader.SequenceReader import readContigs
from vae.VAE_Test import get_input
from writer.BinWriter import mapBinAndContigNames, writeBins


def run_deep_clustering():
    save_dir = 'results/dvmb_model2'
    batch_size, n_epoch = 256, 300
    n_hidden = 64
    destroy = False
    n_clusters = 37

    tnf, rpkm = get_input(batch_size, destroy, save_dir)
    x = sklearn.preprocessing.minmax_scale(tnf, feature_range=(0, 1), axis=1, copy=True)
    x1 = sklearn.preprocessing.minmax_scale(rpkm, feature_range=(0, 1), axis=0, copy=True)

    print(f'x_max={np.max(x)}, x_min={np.min(x)}')
    print(f'x1_max={np.max(x1)}, x1_min={np.min(x1)}')

    inputs = []
    for idx, x in enumerate(x):
        tmp = np.append(x1[idx], x)
        inputs.append(tmp)
    x = np.array(inputs)

    dvmb: DVMB2 = DVMB2(n_hidden=n_hidden, batch_size=batch_size, n_epoch=n_epoch, n_clusters=n_clusters,
                                  save_dir=save_dir)
    dvmb.compile()

    # pre-training
    dvmb.init_vae(x=x)
    # use pre-trained weights
    # dvmb.init_vae(x=x, vae_weights=f'{save_dir}/pretrain_vae_model.h5')
    # real training
    dvmb.fit(x=x, batch_size=batch_size, maxiter=1000)
    # dvmb.load_weights(weights_path=f'{save_dir}/dcec_model_final.h5')

    # predict
    print(f'DVMB loss labels {dvmb.model.metrics_names}')
    clusters = dvmb.predict(x=x)

    # save to bins
    fasta = '/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'
    fastaDict = readContigs(fasta, onlySequence=False)
    binsDict = mapBinAndContigNames(fastaDict, clusters)
    Path(f'{save_dir}/bins').mkdir(exist_ok=True)
    writeBins(f'{save_dir}/bins', bins=binsDict, fastadict=fastaDict)
    print(f'predict size: ', len(clusters))


if __name__ == "__main__":
    run_deep_clustering()
