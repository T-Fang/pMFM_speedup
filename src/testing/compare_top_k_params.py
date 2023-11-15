import sys

sys.path.insert(1, '/home/ftian/storage/pMFM_speedup/')
from src.testing.testing_lib import *


def plot_all_top_k_prediction_vs_actual_cost():
    split_name = 'validation'
    k = 10000
    for group_index in range(1, 15):
        group_index = str(group_index)
        all_costs_prediction_vs_actual_cost(split_name, group_index, k=k)


def compare_top_k_params():
    # plot_all_top_k_prediction_vs_actual_cost()
    # split_name = sys.argv[1]
    # group_index = sys.argv[2]

    k = 10000
    split_name = 'test'
    dataset = load_split(split_name)
    model = load_naive_net_no_SC()
    # model = load_gcn_with_mlp()

    for group_index, subject_group in enumerate(dataset.subject_groups, 1):
        print(f'Testing {split_name} {group_index}')
        group_index = str(group_index)
        get_top_k_prediction(subject_group, model, k=k, ignore_negative_costs=False)
        # gnn_get_top_k_prediction(subject_group, model, k=k, ignore_negative_costs=False)

    pred_and_actual_cost_corr_dist('test',
                                   base_dir=os.path.join(PATH_TO_TESTING_REPORT, 'compare_top_k_params'),
                                   prediction_file_name=f'predicted_top_{k}.csv')


if __name__ == '__main__':
    compare_top_k_params()
    # pass
