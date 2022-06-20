# Deep Variational Metagenomics Binner (DVMB)


## Usage
1. Setup
 - Setup miniconda
 - Create two environments: dmb-conda-env.yml and tools-conda-env.yml
   - dmb-conda-env.yml: create *dmb* profile for developement and runtime
   - tools-conda-env.yml: create *tools* profile for other metagenomics tools
2. Dataset
 - Download via: TBD
3. Run with CAMI Low dataset
 - We set random seed to 2 and disable GPU to reduce as much as possible the randomness, one might need to check for those at begin of test file.
 - Refer DVMB_Test.py for parameter modifications.