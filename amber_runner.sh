#!/bin/bash
# 3 params:
# + full path to parent of bins dir
# + amber output dir
# + label of model

if [ -z "$3" ]
then
  label='DVMB2'
else
  label=$3
fi
echo $label
source ~/.bash_profile
conda activate tools
rm -rf "/share_data/reports/$2"
mkdir -p "/share_data/reports/$2"
/share_data/amber/src/utils/convert_fasta_bins_to_biobox_format.py $1/bins/*.fna -o /share_data/reports/$2/$2.tsv
amber.py -g /share_data/cami_low/gsa_mapping_with_length.binning  /share_data/reports/$2/$2.tsv -l $label -o /share_data/reports/$2