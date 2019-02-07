LOAD_LOC = '../../datasets/translation/split/'
SAVE_LOC_RECORDS = '../../datasets/translation/records/'
SAVE_LOC_NPY ='../../datasets/translation/npy/'

pattern = {
    'src': 'europarl-v7.sv-en.en{}',
    'target': 'europarl-v7.sv-en.sv{}'
}

DATASET_DEFAULTS = {
    'num_cpus': 8,
    'batch_size': 32
}