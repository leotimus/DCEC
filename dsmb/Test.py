import os
from pathlib import Path

import numpy as np
from vamb.__main__ import calc_tnf, calc_rpkm
from vamb.vambtools import numpy_inplace_maskarray, write_npz
from dsmb.DSMB import DSMB
from reader.SequenceReader import readContigs
from writer.BinWriter import mapBinAndContigNames, writeBins


def get_input(batch_size, destroy, save_dir):
    base_path = os.path.dirname(__file__) + "/data"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    bams_path = [base_path +'/cami_low/bams/RL_S001.bam']
    fasta_path = base_path + '/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'


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
    inputs = []
    for idx, x in enumerate(tnf):
        tmp = np.append(rpkm[idx], x)
        inputs.append(tmp)
    inputs = np.array(inputs)
    return inputs


def run_deep_clustering():
    save_dir = '../results/dsmb'
    batch_size, n_epoch = 256, 100
    n_hidden = 64
    destroy = False
    x = get_input(batch_size, destroy, save_dir)
    dvmb = DSMB(n_hidden=n_hidden, batch_size=256, n_epoch=300, n_clusters=60, save_dir=save_dir)
    dvmb.init_sae(x=x, sae_weights=None)
    dvmb.compile()

    # real training
    dvmb.fit(x=x, batch_size=batch_size)
    dvmb.load_weights(weights_path=f'{save_dir}/dcec_model_final.h5')
    # predict
    clusters = dvmb.predict(x=x, batch_size=batch_size)

    # save to bins
    base_path = os.path.dirname(__file__) + "/data"
    fasta = base_path + '/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'
    fastaDict = readContigs(fasta, onlySequence=False)
    binsDict = mapBinAndContigNames(fastaDict, clusters)
    Path(f'{save_dir}/bins').mkdir(exist_ok=True)
    writeBins(f'{save_dir}/bins', bins=binsDict, fastadict=fastaDict)
    print(f'predict size: ', len(clusters))


if __name__ == "__main__":
    run_deep_clustering()