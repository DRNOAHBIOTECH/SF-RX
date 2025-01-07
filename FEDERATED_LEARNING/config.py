DATA_PATH = 'data/'
RESULT_PATH = 'result/'

DATA_SOURCE = ['drugscom', 'pdr'] # ['drugbank', 'drugscom', 'pdr']

SEED = 42
INPUT_SHAPE = 509
OUTOUT_SHAPE = {'drugscom': 3, 'pdr': 5} # {'drugbank': 3, 'drugscom': 3, 'pdr': 5}

TE_FOLD = {'drugscom': 1, 'pdr': 2} # {'drugbank': 0, 'drugscom': 1, 'pdr': 2}
VAL_FOLD = {'drugscom': 4, 'pdr': 4} # {'drugbank': 4, 'drugscom': 4, 'pdr': 4}

BATCH_SIZE = 512