import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from random import sample
from src.basic.constants import PATH_TO_FIGURES, NUM_GROUP_IN_SPLIT, PATH_TO_PROJECT
from src.basic.param_performance import ParamPerformance, ParamCoefCost
from src.utils.SC_utils import load_group_SC
from seaborn import displot


def get_group_params(split_name, group_index):
    group_index = str(group_index)
    path_to_group_param = os.path.join(PATH_TO_PROJECT, 'dataset_generation/dataset/', split_name, group_index,
                                       f'{split_name}_{group_index}.csv')
    group_params = pd.read_csv(path_to_group_param, header=None)
    return group_params


def get_group_param_coef(split_name, group_index):
    param_coef_path = os.path.join(PATH_TO_PROJECT, 'dataset_generation/dataset/', split_name, group_index,
                                   f'{split_name}_{group_index}_coef.csv')
    param_coef = pd.read_csv(param_coef_path, header=None)
    return param_coef


class SubjectGroup:

    def __init__(self, split_name, group_index, device=None, use_SC_with_diag=False):
        # assert split_name in NUM_GROUP_IN_SPLIT.keys()
        group_index = str(group_index)
        self.split_name = split_name
        self.group_index = group_index
        self.device = device
        self.use_SC_with_diag = use_SC_with_diag
        self.load_group_data(split_name, group_index, use_SC_with_diag)

    def __len__(self):
        return len(self.param_performances)

    def load_group_data(self, split_name, group_index, use_SC_with_diag):

        raw_data = get_group_params(split_name, group_index)
        # assert raw_data.shape[
        #     0] == PARAM_VECTOR_DIM + 4  # (209, ) when we have 68 ROI

        self.SC = load_group_SC(split_name, group_index, use_SC_with_diag)
        print(f'SC loaded for {split_name} {group_index}!')

        self.param_performances = []
        for i in range(raw_data.shape[1]):
            param = raw_data.iloc[4:209, i].to_numpy()
            performance = raw_data.iloc[0:4, i].to_numpy()
            param = torch.tensor(param, dtype=torch.float32, device=self.device)
            performance = torch.tensor(performance, dtype=torch.float32, device=self.device)
            param_performance = ParamPerformance(param, performance)
            self.param_performances.append(param_performance)
        print(f'{len(self.param_performances)} params loaded for {split_name} {group_index}!')

    def get_top_k_params(self, k):
        top_k_params = sorted(self.param_performances, key=lambda x: x.total_cost)[:k]
        top_k_params = [param.to_numpy() for param in top_k_params]
        top_k_params = np.stack(top_k_params, axis=1)
        return top_k_params

    def get_meaningful_params(self):
        return [param for param in self.param_performances if param.is_meaningful]

    def sample_k_params(self, k, use_meaningful_only=False):
        population = self.param_performances if not use_meaningful_only else self.get_meaningful_params()
        sampled_params_with_performances = [(p.param, p.performance) for p in sample(population, k)]
        sampled_params_with_performances = list(zip(*sampled_params_with_performances))
        params = torch.stack(sampled_params_with_performances[0], dim=1)
        performances = torch.stack(sampled_params_with_performances[1], dim=1)
        return params, performances

    def plot_FC_corr_distribution(self):
        # we ignore FC_corr_cost of 10, which means the parameter cannot generate meaningful TC
        meaningful_params = [param.FC_corr_cost for param in self.param_performances if param.FC_corr_cost < 10]
        displot(meaningful_params).set(
            title=f'{self.split_name} {self.group_index}: {len(meaningful_params)} params with meaningful FC corr')

    def plot_FC_L1_distribution(self):
        # we ignore FC_L1_cost of 5, which means the parameter cannot generate meaningful TC
        meaningful_params = [param.FC_L1_cost for param in self.param_performances if param.FC_L1_cost < 5]
        displot(meaningful_params).set(
            title=f'{self.split_name} {self.group_index}: {len(meaningful_params)} params with meaningful FC L1')

    def plot_FCD_KS_distribution(self):
        # we ignore FC_corr_cost of 10, which means the parameter cannot generate meaningful TC
        meaningful_params = [param.FCD_KS for param in self.param_performances if param.FCD_KS < 10]
        displot(meaningful_params).set(
            title=f'{self.split_name} {self.group_index}: {len(meaningful_params)} params with meaningful FCD KS')

    def generate_param_distribution(self,
                                    path_to_fig_folder: str = os.path.join(PATH_TO_FIGURES, 'param_distribution/')):
        self.plot_FC_corr_distribution()
        plt.savefig(os.path.join(path_to_fig_folder, 'FC_corr_distribution/',
                                 f'{self.split_name}_{self.group_index}_FC_corr_distribution.png'),
                    bbox_inches='tight')
        plt.close()

        self.plot_FC_L1_distribution()
        plt.savefig(os.path.join(path_to_fig_folder, 'FC_L1_distribution/',
                                 f'{self.split_name}_{self.group_index}_FC_L1_distribution.png'),
                    bbox_inches='tight')
        plt.close()

        self.plot_FCD_KS_distribution()
        plt.savefig(os.path.join(path_to_fig_folder, 'FCD_KS_distribution/',
                                 f'{self.split_name}_{self.group_index}_FCD_KS_distribution.png'),
                    bbox_inches='tight')
        plt.close()


def generate_all_param_distribution(path_to_fig_folder: str = os.path.join(PATH_TO_FIGURES, 'param_distribution/')):
    for split_name, idx_in_split_range in NUM_GROUP_IN_SPLIT.items():
        for idx_in_split in range(1, idx_in_split_range + 1):
            subject_group = SubjectGroup(split_name, idx_in_split)
            subject_group.generate_param_distribution(path_to_fig_folder=path_to_fig_folder)


class ParamCoefSubjectGroup(SubjectGroup):

    def load_group_data(self, split_name, group_index, use_SC_with_diag):
        param_coef = get_group_param_coef(split_name, group_index)

        self.SC = load_group_SC(split_name, group_index, use_SC_with_diag)
        print(f'SC loaded for {split_name} {group_index}!')

        self.param_performances = []
        for i in range(param_coef.shape[1]):
            param = param_coef.iloc[4:, i].to_numpy()
            costs = param_coef.iloc[0:4, i].to_numpy()
            param = torch.tensor(param, dtype=torch.float32, device=self.device)
            costs = torch.tensor(costs, dtype=torch.float32, device=self.device)
            param_coef_cost = ParamCoefCost(param, costs)
            self.param_performances.append(param_coef_cost)
        print(f'{len(self.param_performances)} param coefficients loaded for {split_name} {group_index}!')


if __name__ == '__main__':
    generate_all_param_distribution()
