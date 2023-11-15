import os

# File Names
PATH_TO_PROJECT = '/home/ftian/storage/pMFM_speedup/'
PATH_TO_DATASET = os.path.join(PATH_TO_PROJECT, 'dataset_generation/dataset/')
PATH_TO_PROCESSED_DATA = os.path.join(PATH_TO_PROJECT, 'data/processed/')
PATH_TO_TRAINING_LOG = os.path.join(PATH_TO_PROJECT, 'reports/training_log/')
PATH_TO_FIGURES = os.path.join(PATH_TO_PROJECT, 'reports/figures/')
PATH_TO_TESTING_REPORT = os.path.join(PATH_TO_PROJECT, 'reports/testing/')

# General constants
MANUAL_SEED = 5
NUM_REGION = 68  # number of ROI
PARAM_VECTOR_DIM = NUM_REGION * 3 + 1  # 205 when we have 68 ROI
NUM_GROUP_IN_SPLIT = {'train': 57, 'validation': 14, 'test': 17}
SPLIT_NAMES = ['train', 'validation', 'test']

# Upper Bound for costs
FC_CORR_UPPER_BOUND = 1
FC_L1_UPPER_BOUND = 1
FCD_KS_UPPER_BOUND = 1
TOTAL_COST_UPPER_BOUND = FC_CORR_UPPER_BOUND + FC_L1_UPPER_BOUND + FCD_KS_UPPER_BOUND
# Logging
LOG_INTERVAL = 50
