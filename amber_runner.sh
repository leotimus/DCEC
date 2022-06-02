#!/bin/bash
# 3 params:
# + full path to parent of bins dir
# + amber output dir
# + label of model

if [ -z "$2" ]
then
  label='DVMB'
else
  label=$2
fi
echo $label
source ~/.bash_profile
conda activate tools
mkdir -p "$1/amber"
/share_data/amber/src/utils/convert_fasta_bins_to_biobox_format.py $1/bins/*.fna -o $1/amber.tsv
amber.py -g /share_data/cami_low/gsa_mapping_with_length.binning  $1/amber.tsv -l $label -o $1/amber