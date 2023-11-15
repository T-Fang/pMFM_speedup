from matplotlib import pyplot as plt
from src.basic.constants import NUM_GROUP_IN_SPLIT
from src.basic.subject_group import ParamCoefSubjectGroup, SubjectGroup
from torch.utils.data import Dataset


class ParamDataset(Dataset):

    def __init__(self, split_name: str, device=None):
        # assert split_name in NUM_GROUP_IN_SPLIT.keys()
        self.split_name = split_name
        self.device = device

        self.subject_groups = []
        self.SCs = []
        self.params = []
        self.cost_vectors = []

        self._add_subject_from_split(split_name)

    def _get_subject_group(self, split_name, group_index):
        return SubjectGroup(split_name, group_index, self.device)

    def _add_subject_from_split(self, split_name: str):
        num_of_groups = NUM_GROUP_IN_SPLIT[split_name]

        for i in range(1, num_of_groups + 1):
            subject_group = self._get_subject_group(split_name, i)
            self.subject_groups.append(subject_group)

            for param_performance in subject_group.param_performances:
                self.SCs.append(subject_group.SC)
                self.params.append(param_performance.param)
                self.cost_vectors.append(param_performance.cost_vector)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        return ((self.SCs[index], self.params[index]), self.cost_vectors[index])


class ParamDatasetNoSC(ParamDataset):

    def __getitem__(self, index):
        return (self.params[index], self.cost_vectors[index])


class ParamCoefDataset(ParamDataset):

    def _get_subject_group(self, split_name, group_index):
        return ParamCoefSubjectGroup(split_name, group_index, self.device)