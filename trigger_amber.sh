#!/bin/bash

env='tools'
bins_path='/home/ltms/Projects/DCEC/results/debug2/bins/*.fna'
out_file='dcec_amber/dcec-bins.tsv'
meta_bat2_out='output/metabat2/bin.tsv'
vamb_out='output/vamb_bams/bin.tsv'
output='dcec_amber/validation_0205'

rm -rf $out_file
rm -rf $output
source /share_data/softwares/miniconda/etc/profile.d/conda.sh
cd /share_data/cami_low
conda activate $env
python /share_data/amber/src/utils/convert_fasta_bins_to_biobox_format.py $bins_path -o $out_file
echo "trigger amber.py -g gsa_mapping_with_length.binning -l 'DCEC, metaBAT2, VAMB' $out_file $meta_bat2_out $vamb_out -o $output"
#amber.py -g gsa_mapping_with_ length.binning -l 'DCEC, metaBAT2, VAMB' $out_file $meta_bat2_out $vamb_out -o $output