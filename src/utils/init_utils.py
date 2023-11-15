import os
import pickle
import torch
# import random
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import seed_everything
from torch.utils.data import random_split
from src.basic.constants import MANUAL_SEED, PATH_TO_PROCESSED_DATA
from src.basic.param_dataset import ParamCoefDataset, ParamDataset, ParamDatasetNoSC


def seed_all():
    """
    Automatically seeds across all dataloader workers and processes for torch, numpy and stdlib random number generators.
    """
    # random.seed(MANUAL_SEED)
    # torch.manual_seed(MANUAL_SEED)
    # np.random.seed(MANUAL_SEED)
    seed_everything(MANUAL_SEED)


def set_gpu_device(gpu_number=0):
    device = torch.device('cpu')
    run_on_gpu = False
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_number}')
        run_on_gpu = True
        # torch.cuda.set_device(gpu_number)
        torch.cuda.current_device()
    print(f'Number of cuda devices: {torch.cuda.device_count()}')
    print(f'My device: {device}')
    return device, run_on_gpu


def load_split(split_name: str, use_SC=True, use_coef=False):
    """
    Loads a split of the dataset.
    """
    file_prefix = 'use_coef_' if use_coef else ''
    file_suffix = '' if use_SC else '_no_SC'
    dataset_path = os.path.join(PATH_TO_PROCESSED_DATA, f'{file_prefix}{split_name}_ds{file_suffix}.pickle')
    try:
        dataset = pickle.load(open(dataset_path, "rb"))
    except OSError:
        if use_coef:
            dataset = ParamCoefDataset(split_name)
        elif use_SC:
            dataset = ParamDataset(split_name)
        else:
            dataset = ParamDatasetNoSC(split_name)
        pickle.dump(dataset, open(dataset_path, "wb"))

    print(split_name, 'dataset loaded!')
    return dataset


def load_dataset(use_SC=True, use_coef=False):
    train_ds = load_split('train', use_SC=use_SC, use_coef=use_coef)
    validation_ds = load_split('validation', use_SC=use_SC, use_coef=use_coef)
    test_ds = load_split('test', use_SC=use_SC, use_coef=use_coef)
    return train_ds, validation_ds, test_ds


def shrink_dataset(ds, target_size_ratio):
    print('Size of the original dataset', len(ds))
    target_size = int(len(ds) * target_size_ratio)
    shrunk_dataset, _ = random_split(ds, [target_size, len(ds) - target_size])
    print(f'Using {len(shrunk_dataset)} samples')

    return shrunk_dataset


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
