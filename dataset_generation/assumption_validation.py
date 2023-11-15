import os
from pathlib import Path
from posixpath import split
from random import sample
import sys
import torch
from matplotlib import pyplot as plt
from seaborn import histplot
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')

from src.utils.CBIG_pMFM_basic_functions_HCP import CBIG_combined_cost_train, CBIG_mfm_multi_simulation, csv_matrix_read, bold2fcd, bold2fc, get_ks_cost_between, get_fc_corr_between
from src.utils.pMFM_utils import forward_simulation, get_simulated_fc_and_fcd, get_FC_corr_between_FCs, get_FCD_KS_between_FCDs, extract_param
from src.utils.SC_utils import corr_between_matrices, get_path_to_group, get_path_to_group_SC, get_triu_np_vector, matrix_to_np
from src.utils.init_utils import load_split
from src.basic.cost_type import CostType
from src.basic.constants import PATH_TO_PROJECT, PATH_TO_DATASET, NUM_GROUP_IN_SPLIT
from src.basic.subject_group import SubjectGroup


##########################################################
# costs vs param vector correlation
##########################################################
def plot_cost_vs_param_correlation(cost_type, split_name, group_index):
    base_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_param_correlation/')
    save_dir = os.path.join(base_dir, split_name, group_index)

    all_cost = pd.read_csv(os.path.join(save_dir, f'{cost_type.name}_with_diff_param.csv'), header=None)
    n = all_cost.shape[0]
    all_cost_vector = get_triu_np_vector(all_cost)
    all_cost_series = pd.Series(all_cost_vector)

    param_correlations = pd.read_csv(os.path.join(base_dir, 'corr_between_all_params.csv'), header=None)
    param_correlations = param_correlations.iloc[:n, :n]
    param_correlations_vector = get_triu_np_vector(param_correlations)
    param_correlations_series = pd.Series(param_correlations_vector)

    # Remove NaN values
    valid_cost_indices = all_cost_series.notna() & np.isfinite(all_cost_series)
    all_cost_series = all_cost_series[valid_cost_indices]
    param_correlations_series = param_correlations_series[valid_cost_indices]

    assert all_cost_series.shape[0] == param_correlations_series.shape[0]
    corr = all_cost_series.corr(param_correlations_series)

    cost_and_param_correlation = pd.DataFrame({
        cost_type.name: all_cost_series,
        'param_correlation': param_correlations_series
    })
    cost_and_param_correlation.to_csv(os.path.join(save_dir, f'{cost_type.name}_and_param_correlation.csv'),
                                      index=False)
    cost_and_param_correlation.plot(x='param_correlation',
                                    y=cost_type.name,
                                    kind='scatter',
                                    figsize=(20, 10),
                                    fontsize=16)

    plt.title(f'{cost_type.name} vs param vector correlation (r={corr:.4f})', fontsize=19)
    plt.xlabel(
        f'param vector correlation (mean={param_correlations_series.mean():.4f}, std={param_correlations_series.std():.4f})',
        fontsize=16)
    plt.ylabel(f'{cost_type.name} (mean={all_cost_series.mean():.4f}, std={all_cost_series.std():.4f})', fontsize=16)
    plt.savefig(os.path.join(save_dir, f'{cost_type.name}_vs_param_correlation.png'), dpi=400)


def get_costs_with_diff_param(split_name, group_index):

    all_simulated_FC, all_simulated_FCD = get_all_simulation_with_diff_param(split_name, group_index)
    save_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_param_correlation', split_name, group_index)

    all_FCD_KS = get_FCD_KS_between_FCDs(all_simulated_FCD)
    save_np_array(all_FCD_KS, save_dir, 'FCD_KS_with_diff_param.csv')

    all_FC_CORR, all_FC_L1 = get_FC_corr_between_FCs(all_simulated_FC)
    save_np_array(all_FC_CORR, save_dir, 'FC_CORR_with_diff_param.csv')
    save_np_array(all_FC_L1, save_dir, 'FC_L1_with_diff_param.csv')


def get_all_simulation_with_diff_param(split_name, group_index):
    base_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_param_correlation/', split_name, group_index)
    FC_path = os.path.join(base_dir, 'simulated_FC')
    FCD_path = os.path.join(base_dir, 'simulated_FCD')

    FC_filenames = [f for f in os.listdir(FC_path) if f.endswith('.csv') and f.startswith('simulated_FC')]
    FCD_filenames = [f for f in os.listdir(FCD_path) if f.endswith('.csv') and f.startswith('simulated_FCD')]
    all_simulated_FC = []
    all_simulated_FCD = []

    for filename in FC_filenames:
        simulated_FC_path = os.path.join(FC_path, filename)
        simulated_FC = np.loadtxt(simulated_FC_path, delimiter=',')
        all_simulated_FC.append(simulated_FC)
    for filename in FCD_filenames:
        simulated_FCD_path = os.path.join(FCD_path, filename)
        simulated_FCD = np.loadtxt(simulated_FCD_path, delimiter=',')
        all_simulated_FCD.append(simulated_FCD)

    return all_simulated_FC, all_simulated_FCD


def simulate_with_diff_param(split_name, group_index):
    base_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_param_correlation/')
    fcd_save_dir = os.path.join(base_dir, split_name, group_index, 'simulated_FCD')
    fc_save_dir = os.path.join(base_dir, split_name, group_index, 'simulated_FC')

    # get param vector
    param_vectors = pd.read_csv(os.path.join(base_dir, 'params.csv'), header=None).to_numpy()
    param_vectors = torch.from_numpy(param_vectors).type(torch.DoubleTensor)

    for param_idx in range(param_vectors.shape[1]):
        param_vector = param_vectors[:, param_idx].unsqueeze(1)
        print(param_vector.shape)
        fc, fcd_cdf = get_simulated_fc_and_fcd(split_name, group_index, param_vector)
        save_np_array(fcd_cdf, fcd_save_dir, f'simulated_FCD_{param_idx}.csv')
        save_np_array(fc, fc_save_dir, f'simulated_FC_{param_idx}.csv')


def get_corr_between_all_params():
    save_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_param_correlation/')

    # get param vector
    param_vectors = pd.read_csv(os.path.join(save_dir, 'params.csv'), header=None).to_numpy()
    corr_matrix = np.corrcoef(param_vectors, rowvar=False)
    save_np_array(corr_matrix, save_dir, 'corr_between_all_params.csv')


def pick_random_params(split_name='validation',
                       n_params_per_group=10,
                       save_dir=os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_param_correlation')):

    def get_random_params_from(group_index):
        path_to_candidate_params = os.path.join(PATH_TO_DATASET, split_name, group_index, 'top_100_params.csv')
        candidate_params = pd.read_csv(path_to_candidate_params, header=None).to_numpy()
        sample_indices = sample(range(candidate_params.shape[1]), n_params_per_group)
        random_params = candidate_params[4:, sample_indices]
        random_params = torch.from_numpy(random_params).type(torch.DoubleTensor)
        return random_params

    random_params_list = []
    for group_index in range(1, NUM_GROUP_IN_SPLIT[split_name] + 1):
        group_index = str(group_index)
        random_params_list.append(get_random_params_from(group_index))

    all_random_params = np.concatenate(random_params_list, axis=1)
    save_np_array(all_random_params, save_dir, 'params.csv')
    return all_random_params


##########################################################
# Costs vs SC correlation
##########################################################


def plot_cost_vs_SC_correlation(cost_type: CostType, param_folder_name):
    save_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_SC_correlation')

    all_cost = pd.read_csv(os.path.join(save_dir, param_folder_name, f'{cost_type.name}_with_diff_SC.csv'), header=None)
    n = all_cost.shape[0]
    all_vectorized_cost = get_triu_np_vector(all_cost)
    all_cost_series = pd.Series(all_vectorized_cost)

    SC_correlations = pd.read_csv(os.path.join(save_dir, 'corr_between_all_SCs.csv'), header=None)
    SC_correlations = SC_correlations.iloc[:n, :n]
    SC_correlations_vector = get_triu_np_vector(SC_correlations)
    SC_correlations_series = pd.Series(SC_correlations_vector)

    # Remove NaN values
    valid_cost_indices = all_cost_series.notna() & np.isfinite(all_cost_series)
    all_cost_series = all_cost_series[valid_cost_indices]
    SC_correlations_series = SC_correlations_series[valid_cost_indices]

    assert all_cost_series.shape[0] == SC_correlations_series.shape[0]
    corr = all_cost_series.corr(SC_correlations_series)

    cost_and_SC_correlation = pd.DataFrame({cost_type.name: all_cost_series, 'SC_correlation': SC_correlations_series})
    cost_and_SC_correlation.to_csv(os.path.join(save_dir, param_folder_name,
                                                f'{cost_type.name}_and_SC_correlation.csv'),
                                   index=False)
    cost_and_SC_correlation.plot(x='SC_correlation', y=cost_type.name, kind='scatter', figsize=(20, 10), fontsize=16)

    plt.title(f'{cost_type.name} vs SC correlation (r={corr:.4f})', fontsize=19)
    plt.xlabel(f'SC correlation between subject groups (mean={SC_correlations_series.mean():.4f}, \
            std={SC_correlations_series.std():.4f})',
               fontsize=18)
    plt.ylabel(f'{cost_type.name} (mean={all_cost_series.mean():.4f}, std={all_cost_series.std():.4f})', fontsize=18)
    plt.savefig(os.path.join(save_dir, param_folder_name, f'{cost_type.name}_vs_SC_correlation.png'), dpi=400)


def get_costs_with_diff_SC(param_folder_name):
    split_names = ['train', 'validation']

    all_simulated_FCD = []
    all_simulated_FC = []
    for split_name in split_names:
        simulated_FCs, simulated_FCDs = get_all_simulation_with_diff_SC(split_name, param_folder_name)
        all_simulated_FC.extend(simulated_FCs)
        all_simulated_FCD.extend(simulated_FCDs)

    save_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_SC_correlation', param_folder_name)

    # Get FCD_KS between FCDs simulated with different SCs
    all_FCD_KS = get_FCD_KS_between_FCDs(all_simulated_FCD)
    save_np_array(all_FCD_KS, save_dir, 'FCD_KS_with_diff_SC.csv')

    # Get FC_CORR and FC_L1 between FCs simulated with different SCs
    all_FC_CORR, all_FC_L1 = get_FC_corr_between_FCs(all_simulated_FC)
    save_np_array(all_FC_CORR, save_dir, 'FC_CORR_with_diff_SC.csv')
    save_np_array(all_FC_L1, save_dir, 'FC_L1_with_diff_SC.csv')


def get_all_simulation_with_diff_SC(split_name, param_folder_name):
    all_simulated_FC = []
    all_simulated_FCD = []
    for group_index in range(1, NUM_GROUP_IN_SPLIT[split_name] + 1):
        group_index = str(group_index)
        group_path = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_SC_correlation', param_folder_name,
                                  split_name, group_index)
        simulated_FC = np.loadtxt(os.path.join(group_path, 'simulated_FC.csv'), delimiter=',')
        simulated_FCD = np.loadtxt(os.path.join(group_path, 'simulated_FCD.csv'), delimiter=',')
        all_simulated_FC.append(simulated_FC)
        all_simulated_FCD.append(simulated_FCD)

    return all_simulated_FC, all_simulated_FCD


def simulate_all_with_diff_SC(param_folder_name):
    split_names = ['train', 'validation']
    for split_name in split_names:
        for group_index in range(1, NUM_GROUP_IN_SPLIT[split_name] + 1):
            group_index = str(group_index)
            simulate_with_diff_SC(split_name, group_index, param_folder_name)


def simulate_with_diff_SC(split_name, group_index, param_folder_name):
    save_dir = os.path.join(PATH_TO_PROJECT, 'dataset_generation/costs_vs_SC_correlation', param_folder_name)

    # get param vector
    param_vector = pd.read_csv(os.path.join(save_dir, 'param.csv'), header=None).to_numpy()
    param_vector = torch.from_numpy(param_vector).type(torch.DoubleTensor)

    fc, fcd_cdf = get_simulated_fc_and_fcd(split_name, group_index, param_vector)

    save_dir = os.path.join(save_dir, split_name, group_index)
    save_np_array(fcd_cdf, save_dir, 'simulated_FCD.csv')
    save_np_array(fc, save_dir, 'simulated_FC.csv')


##########################################################
# Use the same parameter on different SCs
##########################################################
def test_swap_param(swapping_subject_group,
                    swapped_subject_group,
                    path_to_swapped_group,
                    num_param_to_swap=10,
                    file_name: str = 'simulation_with_swapped_params.csv'):
    SC_corr = corr_between_matrices([swapping_subject_group.SC, swapped_subject_group.SC])[0][1]
    print(f'SC corr between {swapping_subject_group.split_name} {swapping_subject_group.group_index} \
            and {swapped_subject_group.split_name} {swapped_subject_group.group_index} is {SC_corr}')
    params, original_performances = swapping_subject_group.sample_k_params(num_param_to_swap)
    params = params.cpu().numpy()
    original_performances = original_performances.cpu().numpy()
    # print('PARAM VECTORS:', params.shape)  # shape: (205, k)
    # print('ORIGINAL PERFORMANCE:', original_performances.shape)  # shape: (4, k)

    fc_corr_cost, fc_L1_cost, fcd_cost, total_cost = forward_simulation(params, path_to_swapped_group)
    new_performances = np.stack((fc_corr_cost, fc_L1_cost, fcd_cost, total_cost), axis=0)  # shape: (4, k)
    performance_diff = new_performances - original_performances
    SC_corr_row = np.full((1, num_param_to_swap), SC_corr)  # shape: (1, k)

    result_after_swapping = np.concatenate(
        (SC_corr_row, performance_diff, original_performances, new_performances, params), axis=0)
    # print('Result after swapping:', result_after_swapping.shape)  # shape: (218, k)
    np.savetxt(file_name, result_after_swapping, delimiter=',')
    check_swap_param_effect(file_name)

    return result_after_swapping


def check_swap_param_effect(path_to_file):
    df = pd.read_csv(path_to_file, header=None)
    SC_corr = df.iloc[0, 0]
    print("correlation between two SCs:", SC_corr)
    n = df.shape[1]
    print(f'{n} parameters have been swapped')

    total_cost_diff = df.iloc[4, :]
    much_worse_count = total_cost_diff.where(lambda x: x > 2).count()
    print(f'{much_worse_count} parameters with new SC does not have meaningful time course while the original one has')
    much_better_count = total_cost_diff.where(lambda x: x < -2).count()
    print(
        f'{much_better_count} parameters with the original SC does not have meaningful time course while the new one has'
    )
    both_not_meaningful_count = total_cost_diff.where(lambda x: x == 0).count()
    print(
        f'{both_not_meaningful_count} parameters with identical performance after swapping (most likely due to no meaningful time course being generated in either case)'  # noqa E501
    )
    both_meaningful_count = len(total_cost_diff) - much_worse_count - much_better_count - both_not_meaningful_count
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    fig.suptitle(
        f'Cost differences, where SC\'s correlation is {SC_corr:.6f} ({both_meaningful_count} params involved)',
        y=1.08,
        fontsize=18)

    # plot total cost difference
    total_cost_diff = total_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(total_cost_diff, ax=ax[0]).set_title(
        f'Total cost difference\n(mean={total_cost_diff.mean():.4f}, std={total_cost_diff.std():.4f})')
    ax[0].set_xlabel('total cost difference')

    # plot FC_CORR cost difference
    fc_corr_cost_diff = df.iloc[1, :]
    fc_corr_cost_diff = fc_corr_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(fc_corr_cost_diff, ax=ax[1]).set_title(
        f'FC_CORR cost difference\n(mean={fc_corr_cost_diff.mean():.4f}, std={fc_corr_cost_diff.std():.4f})')
    ax[1].set_xlabel('FC_CORR cost difference')

    # plot FC_L1 cost difference
    fc_L1_cost_diff = df.iloc[2, :]
    fc_L1_cost_diff = fc_L1_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(fc_L1_cost_diff, ax=ax[2]).set_title(
        f'FC_L1 cost difference\n(mean={fc_L1_cost_diff.mean():.4f}, std={fc_L1_cost_diff.std():.4f})')
    ax[2].set_xlabel('FC_L1 cost difference')

    # plot FCD_KS cost difference
    fcd_cost_diff = df.iloc[3, :]
    fcd_cost_diff = fcd_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(
        fcd_cost_diff,
        ax=ax[3]).set_title(f'FCD cost difference\n(mean={fcd_cost_diff.mean():.4f}, std={fcd_cost_diff.std():.4f})')
    ax[3].set_xlabel('FCD cost difference')

    img_file_path = os.path.splitext(path_to_file)[0] + '.png'
    fig.savefig(img_file_path, bbox_inches='tight')


def swap_param_wrapper():
    path_to_swapped_group = os.path.join(PATH_TO_PROJECT, 'dataset_generation/input_to_pMFM/train/1')
    # swapped_subject_group = SubjectGroup('train', 1)
    swapping_subject_group = SubjectGroup('train', 1)

    split_name = sys.argv[1]
    group_index = sys.argv[2]
    swapped_subject_group = SubjectGroup(split_name, group_index)

    test_swap_param(
        swapping_subject_group,
        swapped_subject_group,
        path_to_swapped_group,
        num_param_to_swap=5000,
        file_name=os.path.join(
            PATH_TO_PROJECT,
            # f'dataset_generation/swap_params/use_{split_name}_{group_index}_param_on_train_1_SC.csv'))
            f'dataset_generation/swap_params/use_train_1_param_on_{split_name}_{group_index}_SC.csv'))


def save_np_array(array, save_dir, file_name):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(save_dir, file_name), array, delimiter=',')


def debug_simulation():
    param = pd.read_csv('param_2.csv', header=None).iloc[:, 0]
    # param = extract_param('train', '1', 0)

    param = matrix_to_np(param)
    param = np.expand_dims(param, axis=1)
    path_to_group = get_path_to_group('train', '1')
    fc_corr_cost, fc_L1_cost, fcd_cost, total_cost = forward_simulation(param, path_to_group)
    print(fc_corr_cost, fc_L1_cost, fcd_cost, total_cost)


if __name__ == '__main__':
    # swap_param_wrapper()
    # get_simulated_fcd_with_same_param()
    debug_simulation()