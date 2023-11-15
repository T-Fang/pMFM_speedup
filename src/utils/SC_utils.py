import os
from sysconfig import get_path
import torch
from src.basic.constants import SPLIT_NAMES, NUM_GROUP_IN_SPLIT, PATH_TO_PROJECT
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_path_to_group(split_name, group_index):
    group_index = str(group_index)
    path_to_pMFM_input = os.path.join(PATH_TO_PROJECT, 'dataset_generation/input_to_pMFM/')
    path_to_group = os.path.join(path_to_pMFM_input, split_name, group_index)
    return path_to_group


def get_path_to_group_SC(split_name, group_index, use_SC_with_diag=False):
    path_to_group = get_path_to_group(split_name, group_index)
    file_name = 'group_level_SC_with_diag.csv' if use_SC_with_diag else 'group_level_SC.csv'
    path_to_group_SC = os.path.join(path_to_group, file_name)
    return path_to_group_SC


def load_group_SC(split_name, group_index, use_SC_with_diag=False):
    SC_path = get_path_to_group_SC(split_name, group_index, use_SC_with_diag)
    SC = pd.read_csv(SC_path, header=None)
    return df_to_tensor(SC)


def get_all_group_SC():
    all_group_SC = []
    for split_name in SPLIT_NAMES:
        all_group_SC.extend(get_SC_in_split(split_name))
    return all_group_SC


def get_SC_in_split(split_name):
    SCs_in_split = []
    for group_index in range(1, NUM_GROUP_IN_SPLIT[split_name] + 1):
        print(f'Loading SC for {split_name} {group_index}')
        group_SC = load_group_SC(split_name, group_index)
        SCs_in_split.append(group_SC)
    return SCs_in_split


def df_to_tensor(df: DataFrame, device=None):
    np_df = matrix_to_np(df)
    df_tensor = torch.tensor(np_df, dtype=torch.float32, device=device)
    return df_tensor


def matrix_to_np(matrix):
    if isinstance(matrix, DataFrame):
        matrix = matrix.to_numpy()
    elif isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    elif isinstance(matrix, Series):
        matrix = matrix.to_numpy()
    return matrix


def get_triu_np_vector(matrix):
    np_matrix = matrix_to_np(matrix)
    n = np_matrix.shape[0]
    triu_vector = np_matrix[np.triu_indices(n, 1)]
    return triu_vector


def get_triu_torch_vector(matrix: torch.Tensor):
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, 1)]


def batched_SC_to_triu_vector(batched_SC: torch.Tensor):
    return torch.stack([get_triu_torch_vector(SC) for SC in batched_SC])


def corr_between_matrices(matrices):
    matrices_triu_vectors = [get_triu_np_vector(matrix) for matrix in matrices]
    return np.corrcoef(matrices_triu_vectors)


def show_heatmap(matrix, filename=None):
    sns.heatmap(matrix)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, dpi=400)
        plt.imshow(plt.imread(filename))


def show_corr_between_all_SCs(filename=f'{PATH_TO_PROJECT}reports/figures/corr_between_all_SCs.png'):
    SCs = get_all_group_SC()
    corr_matrix = corr_between_matrices(SCs)
    np.savetxt('./corr_between_all_SCs.csv', corr_matrix, delimiter=',')
    show_heatmap(corr_matrix, filename=filename)
