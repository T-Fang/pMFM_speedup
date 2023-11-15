import sys

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')
from src.training.training_lib import *

if __name__ == '__main__':
    save_dir = os.path.join(PATH_TO_TRAINING_LOG, "basic_models/naive_net/with_SC_use_coef/")
    tune(objective_naive_net_with_SC_use_coef, n_trials=100, timeout=208800, save_dir=save_dir, use_coef=True)
