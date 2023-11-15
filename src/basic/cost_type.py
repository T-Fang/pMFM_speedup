from enum import Enum


class CostType(Enum):
    FC_CORR = 1
    FC_L1 = 2
    FCD_KS = 3

    @staticmethod
    def num_of_cost_type():
        return 3

    def __str__(self):
        if self == CostType.FC_CORR:
            return 'FC_CORR'
        elif self == CostType.FC_L1:
            return 'FC_L1'
        else:
            return 'FCD_KS'
