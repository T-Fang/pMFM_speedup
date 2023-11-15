import warnings
import sys

from dataset_generation import generate_parameters_for
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    split_name = sys.argv[1]
    idx_in_split = sys.argv[2]
    idx_in_split = int(idx_in_split)
    generate_parameters_for(split_name, [idx_in_split])
