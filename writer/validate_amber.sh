#!/bin/bash
amber/src/utils/convert_fasta_bins_to_biobox_format.py "results/$1/bins/*.fna" -o "results/$1/$1.tsv"
amber.py -g /share_data/cami_low/gsa_mapping_with_length.binning  "results/$1/$1.tsv" -l dvmb -o "results/$1/amber_val"