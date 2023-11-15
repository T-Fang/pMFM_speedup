import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from matplotlib import pyplot as plt
from src.basic.constants import NUM_GROUP_IN_SPLIT, PATH_TO_PROJECT, PATH_TO_PROCESSED_DATA, SPLIT_NAMES
from src.basic.subject_group import SubjectGroup
from src.utils.gnn_utils import edge_from_SC


def get_data_list(split_name, group_index):
    data_list = []
    subject_group = SubjectGroup(split_name, group_index, use_SC_with_diag=True)
    edge_index, edge_weight = edge_from_SC(subject_group.SC)
    for param_performance in subject_group.param_performances:
        x = param_performance.pyg_nodes
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight * param_performance.G,
                    y=param_performance.cost_vector.unsqueeze(0))

        data_list.append(data)
    return data_list


class GnnParamDataset(InMemoryDataset):

    def __init__(self,
                 split_name: str,
                 root=os.path.join(PATH_TO_PROCESSED_DATA, 'gnn'),
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.split_name = split_name

        self.data, self.slices = torch.load(self.path_to_split_file(self.split_name))

    def path_to_split_file(self, split_name: str):
        return self.processed_paths[SPLIT_NAMES.index(split_name)]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['train_ds.pt', 'validation_ds.pt', 'test_ds.pt']

    def download(self):
        pass

    def process(self):
        for split_name in SPLIT_NAMES:
            data_list = []
            num_of_groups = NUM_GROUP_IN_SPLIT[split_name]
            for i in range(1, num_of_groups + 1):
                data_list.extend(get_data_list(split_name, i))

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.path_to_split_file(split_name))


if __name__ == '__main__':
    gnn_param_dataset = GnnParamDataset('train')