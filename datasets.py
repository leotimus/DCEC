from base64 import decode
import numpy as np
import tensorflow

import reader.SequenceReader as sr
import matplotlib.pyplot as plt
import os
from pathlib import Path

from vamb.__main__ import calc_tnf, calc_rpkm
from vamb.vambtools import numpy_inplace_maskarray, write_npz


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float32')
    x /= 2.0
    x = x.reshape([-1, 16, 16, 1])
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_fasta(n_samples=None, contig_len=1000):
    lst = get_sequence_samples(n_samples)
    data = [decode(contig, contig_len) for contig in lst]
    for i in data:
        if i.shape != (contig_len, 4):
            data.remove(i)
    
    x = np.array(data)
    print('FASTA:', x.shape)
    x = x.reshape(-1, contig_len, 4, 1).astype('float32')
    print('FASTA:', x.shape)
    return x, None


def load_tnf_rpkm():
    save_dir, outdir = 'vae/results/vae', 'vae/results/vae'
    batch_size, n_epoch = 100, 500
    n_hidden = 64
    destroy = False

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    bams_path = ['/share_data/cami_low/bams/RL_S001.bam']
    fasta_path = '/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta'
    tnf_path = f'{save_dir}/tnf.npz'
    names_path = f'{save_dir}/names.npz'
    lengths_path = f'{save_dir}/lengths.npz'
    rpkm_path = f'{save_dir}/rpkm.npz'
    with open(f'{save_dir}/vectors.log', 'w') as logfile:
        tnf, contignames, contiglengths = calc_tnf(outdir=save_dir,
                                                   fastapath=None,
                                                   tnfpath=tnf_path,
                                                   namespath=names_path,
                                                   lengthspath=lengths_path,
                                                   mincontiglength=100,
                                                   logfile=logfile)
        if not Path(names_path).exists():
            write_npz(os.path.join(save_dir, 'names.npz'), contignames)
        rpkm = calc_rpkm(outdir=save_dir,
                         bampaths=None,
                         rpkmpath=rpkm_path,
                         jgipath=None, mincontiglength=100, refhash=None, ncontigs=len(tnf), minalignscore=None,
                         minid=None,
                         subprocesses=min(os.cpu_count(), 8), logfile=logfile)

    # return run_vamb_ptorch(batch_size, logfile, outdir, rpkm, tnf)

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
    return inputs, None

def get_sequence_samples(n_samples=None):
    fastaFile = "/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
    contigs = sr.readContigs(fastaFile, numberOfSamples=n_samples)
    print(f'Parsed {len(contigs.keys())} contigs')
    lst = list(contigs.values())
    return lst


def myMapCharsToInteger(data):
  # define universe of possible input values
  seq = 'ACTGO'
  # define a mapping of chars to integers
  char_to_int = dict((c, i) for i, c in enumerate(seq))
  #print("Chars to int")
  #print(char_to_int)
  # integer encode input data
  integer_encoded = [char_to_int[char] for char in data]
  return integer_encoded


def setCorrectSequenceLength(n, size):
    #an ugly fix - sequences that didn't contain O or N were one hot encoded in a shape (length, 4) or (length, 5) which was causing problems with reshaping
    #se we make sure there is at least one N and one O in every sequence
    if len(n) > size:
        return n[:size]
    elif len(n) < size:
        return n.ljust(size, "O")
    return n

def setSequenceLength(n, size):
    if len(n) > size:
        return n[:size]
    elif len(n) < size:
        padding = np.array([[-1., -1., -1., -1.]] *(size - len(n)))
        n = np.concatenate((n, padding), axis=0)
    return n


def decode(n, contig_len=20000):
  """
  decoded = bytes(n).decode()
  most_common_nucleotide = max(set(decoded), key=decoded.count)
  decoded = setCorrectSequenceLength(decoded, 1000)
  decoded = [most_common_nucleotide if x == 'N' else x for x in decoded]
  print(decoded)
  return to_categorical(myMapCharsToInteger(decoded), num_classes=5)
  """
  decoded = bytes(n).decode()
  #most_common_nucleotide = max(set(decoded), key=decoded.count)
  #decoded = [most_common_nucleotide if x == 'N' else x for x in decoded]
  #encodings = tensorflow.keras.utils.to_categorical(myMapCharsToInteger(decoded), num_classes=4)
  #encodings = setSequenceLength(encodings, contig_len)
  return decoded

 
def strLengths(n):
  decoded = bytes(n).decode()
  return len(decoded)