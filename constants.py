# Code constants

MIN_SIDE_EFFECT_FREQUENCY = 500
# If side effect occures lesser than this, do not use triples
# (drug1, drug2, side_effect) with such side effect

# Model parameters
PARAMS = {'neg_sample_size': 1,
              'learning_rate': 0.001,
              'epochs': 50,
              'hidden1': 64,
              'hidden2': 32,
              'weight_decay': 0,
              'dropout': 0.1,
              'max_margin': 0.1,
              'batch_size': 512,
              'bias': True}

INPUT_FILE_PATH = "data/input"