import sys
import os
import torch
from torch.multiprocessing import Pool, set_start_method
from torchensemble import VotingClassifier, BaggingClassifier, GradientBoostingClassifier, FusionClassifier

from src.utils.init_utils import seed_all, set_gpu_device, load_dataset, load_split

from src.basic.constants import PATH_TO_TRAINING_LOG, PATH_TO_FIGURES
from src.basic.subject_group import SubjectGroup
from src.basic.cost_type import CostType

from src.models.extract_SC_feat import ExtractScFeatMLP
from src.models.naive_net import NaiveNet, objective_naive_net_no_SC, objective_naive_net_no_SC_use_coef, objective_naive_net_with_SC, objective_naive_net_with_SC_use_coef
from src.models.gnn.gnn_param_dataset import GnnParamDataset
from src.models.gnn.gcn import Gcn, objective_gcn

from src.utils.training_utils import tune, tune_gnn
from src.utils.test_utils import check_all_good_params_costs_for_models_in, get_top_k_prediction, gnn_get_top_k_prediction, plot_top_k_distribution, all_costs_prediction_vs_actual_cost, pred_and_actual_cost_corr_dist

seed_all()

device, run_on_gpu = set_gpu_device()
num_epochs = 100
