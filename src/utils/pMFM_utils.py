import os
import torch
import numpy as np
import pandas as pd

from src.utils.CBIG_pMFM_basic_functions_HCP import CBIG_combined_cost_train, CBIG_mfm_multi_simulation, csv_matrix_read, bold2fcd, bold2fc, get_ks_cost_between, get_fc_corr_between  # noqa: E501
from src.utils.SC_utils import get_path_to_group_SC
from src.basic.constants import PATH_TO_DATASET


def forward_simulation(param_vectors, path_to_group: str, n_dup: int = 5):
    """
    Args:
        param_vectors:  (N*3+1)*M matrix. 
                        N is the number of ROI
                        M is the number of candidate parameter vectors. 
    """
    FCD = os.path.join(path_to_group, 'group_level_FCD.mat')
    SC = os.path.join(path_to_group, 'group_level_SC.csv')
    FC = os.path.join(path_to_group, 'group_level_FC.csv')
    total_cost, fc_corr_cost, fc_L1_cost, fcd_cost, _, _, _ = CBIG_combined_cost_train(
        param_vectors, n_dup, FCD, SC, FC, 0)
    return fc_corr_cost, fc_L1_cost, fcd_cost, total_cost


def get_simulated_fc_and_fcd(split_name, group_index, param_vectors, n_dup=5):
    """
    Get simulated BOLD signal with the given parameter vectors and SC from the subject group indicated by `split_name` and `group_index` # noqa: E501
    """
    # get group SC matrix
    sc_mat_raw = csv_matrix_read(get_path_to_group_SC(split_name, group_index))
    sc_mat = sc_mat_raw * 0.02 / sc_mat_raw.max()
    sc_mat = torch.from_numpy(sc_mat).type(torch.DoubleTensor)

    if torch.cuda.is_available():
        param_vectors = param_vectors.cuda()
        sc_mat = sc_mat.cuda()

    # Calculating simulated BOLD signal using MFM
    bold_d, S_E_all, S_I_all, r_E_all, J_I = CBIG_mfm_multi_simulation(param_vectors, sc_mat, 14.4, n_dup, 0, 0.006, 0)

    # Calculating simulated FCD
    fcd_cdf = bold2fcd(bold_d, n_dup)
    fc = bold2fc(bold_d, n_dup)

    return fc, fcd_cdf


def get_FC_corr_between_FCs(all_FCs):
    n_FC = len(all_FCs)

    all_FC_CORR = np.zeros((n_FC, n_FC))
    all_FC_L1 = np.zeros((n_FC, n_FC))
    for i in range(n_FC):
        for j in range(i + 1, n_FC):
            fc_i = all_FCs[i]
            fc_j = all_FCs[j]
            FC_CORR, FC_L1 = get_fc_corr_between(fc_i, fc_j, use_corr_cost=False)
            print(f'FC_CORR between #{i} and #{j} is {FC_CORR}, and FC_L1 between #{i} and #{j} is {FC_L1}')
            all_FC_CORR[i, j] = FC_CORR
            all_FC_CORR[j, i] = FC_CORR
            all_FC_L1[i, j] = FC_L1
            all_FC_L1[j, i] = FC_L1

    return all_FC_CORR, all_FC_L1


def get_FCD_KS_between_FCDs(all_FCDs):
    n_FCD = len(all_FCDs)

    all_FCD_KS = np.zeros((n_FCD, n_FCD))
    for i in range(n_FCD):
        for j in range(i + 1, n_FCD):
            fcd_i = all_FCDs[i]
            fcd_j = all_FCDs[j]
            ks_cost = get_ks_cost_between(fcd_i, fcd_j)
            print(f'FCD_KS between #{i} and #{j} is {ks_cost}')
            all_FCD_KS[i, j] = ks_cost
            all_FCD_KS[j, i] = ks_cost

    return all_FCD_KS


def extract_param(split_name, group_index, param_index, save_param=False):
    params_path = os.path.join(PATH_TO_DATASET, split_name, group_index, f'{split_name}_{group_index}.csv')
    df = pd.read_csv(params_path, header=None)
    param_with_cost = df.iloc[:, param_index]
    cost = param_with_cost[:4]
    param = param_with_cost.iloc[4:]
    if save_param:
        param.to_csv('param.csv', header=None, index=None)
        cost.to_csv(f'param_cost_with_{split_name}_{group_index}.csv', header=None, index=None)
    return param
