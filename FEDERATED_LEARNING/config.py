DATA_PATH = 'data/'
RESULT_PATH = 'result/'

DATA_SOURCE = ['drn', 'pdr']

SEED = 42
INPUT_SHAPE = 509
OUTOUT_SHAPE = {'drn': 3, 'pdr': 5}

TE_FOLD = {'drn': 1, 'pdr': 2}
VAL_FOLD = {'drn': 4, 'pdr': 4}

BATCH_SIZE = 512