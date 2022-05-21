from base64 import decode
import numpy as np
import tensorflow

import reader.SequenceReader as sr
import matplotlib.pyplot as plt


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

def create_gold_standard_file(contigKeys):
    #create new gold standard file based on available contigs, set by

    print('Creating new gold standard file....')

    newGoldStandardFile = open('C:/Python/Datasets/gold_standard.binning', 'w')

    newGoldStandardFile.write('@Version:0.9.1' + '\n')    
    newGoldStandardFile.write('@SampleID:gsa' + '\n')    
    newGoldStandardFile.write('\n')
    newGoldStandardFile.write('@@SEQUENCEID BINID   TAXID   _contig_id  _number_reads   _LENGTH' + '\n')

    goldStandardList = np.loadtxt("C:/Python/Datasets/gsa_mapping_with_length.binning", delimiter='\t', dtype=str, skiprows=(4))

    for i in goldStandardList:
        seqID = i[0]

        if seqID in contigKeys:
            newGoldStandardFile.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2])+ '\t' + str(i[3]) + '\t' + str(i[4]) + '\t' + str(i[5]) + '\n')

    newGoldStandardFile.close()
    print('Done...')

def load_fasta(n_samples=None, contig_len=1000):
    lst = get_azolla_samples(n_samples)
    data = [decode(contig, contig_len) for contig in lst]
    for i in data:
        if i.shape != (contig_len, 4):
            data.remove(i)
    
    x = np.array(data)
    print('FASTA:', x.shape)
    x = x.reshape(-1, contig_len, 4).astype(np.float32)
    print('FASTA:', x.shape)
    return x, None

def get_azolla_samples(n_samples=None, min_length=750):
    contigLengths = open('./results/contig_length.txt', 'w')

    fastaFile = "C:/Python/Datasets/azolla.fasta"
    contigs = sr.readContigs(fastaFile, numberOfSamples=n_samples)
    print(f'Parsed {len(contigs.keys())} contigs')
    newContigs = dict()
    for key, value in contigs.items():
        contigLengths.write(str(len(value)) + '\n')
        if len(value)>=min_length:
            newContigs[key] = value
    
    print(len(newContigs.keys()))    
    lst = list(newContigs.values())

    contigLengths.close()
    return lst

def get_sequence_samples(n_samples=None, min_length=750):
    fastaFile = "C:/Python/Datasets/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
    contigs = sr.readContigs(fastaFile, numberOfSamples=n_samples)
    print(f'Parsed {len(contigs.keys())} contigs')
    newContigs = dict()
    for key, value in contigs.items():
        if len(value)>=min_length:
            newContigs[key] = value
    print(len(newContigs.keys()))
    binList = get_bin_sequence(list(newContigs.keys()))
    #create_gold_standard_file(list(newContigs.keys()))
    
    lst = list(newContigs.values())
    return lst, binList

def get_bin_sequence(contigKeys):
    binlist = []
    binValuelist = []
    
    originalList = np.loadtxt("C:/Python/Datasets/gsa_mapping_header.binning", delimiter='\t', dtype=str, skiprows=(1))
    originalList = originalList[:,[0,1]] 
    
    for i in contigKeys:
        binID = np.where(originalList[:,0] == i)
        bin = originalList[binID]
        binValue = bin[0][1]
        
        if binValuelist.count(binValue) > 750:
            binNumber = binValuelist.index(binValue)
        else:
            binValuelist.append(binValue)
            binNumber = binValuelist.index(binValue)
            
        binlist.append(binNumber)

        if i == "RL|S1|C1412":
            print("Contig: ", i)
            print("Original Bin:", binValue)
            print("New Bin ID:", binNumber)
        
    return binlist 
    
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
        padding = np.array([[0., 0., 0., 0.]] *(size - len(n)))
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
  most_common_nucleotide = max(set(decoded), key=decoded.count)
  decoded = [most_common_nucleotide if x == 'N' else x for x in decoded]
  encodings = tensorflow.keras.utils.to_categorical(myMapCharsToInteger(decoded), num_classes=4)
  encodings = setSequenceLength(encodings, contig_len)
  return encodings

 
def strLengths(n):
  decoded = bytes(n).decode()
  return len(decoded)