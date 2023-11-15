import torch
from torch import Tensor
from src.basic.constants import FC_CORR_UPPER_BOUND, FC_L1_UPPER_BOUND, FCD_KS_UPPER_BOUND, NUM_REGION


class ParamPerformance:

    def __init__(self, param: Tensor, performance: Tensor):
        self.param = param
        self.performance = performance
        self.device = param.device
        self.is_meaningful = self.FC_CORR_cost < 10 and self.FC_L1_cost < 5 and self.FCD_KS < 10

        self.clamp_costs()

    def clamp_costs(self):
        self.performance[0] = torch.clamp(self.performance[0], 0, FC_CORR_UPPER_BOUND)
        self.performance[1] = torch.clamp(self.performance[1], 0, FC_L1_UPPER_BOUND)
        self.performance[2] = torch.clamp(self.performance[2], 0, FCD_KS_UPPER_BOUND)
        self.performance[3] = torch.sum(self.performance[:3])

    @property
    def cost_vector(self):
        return self.performance[:3]

    @property
    def wEE(self):
        return self.param[0:NUM_REGION]

    @property
    def wEI(self):
        return self.param[NUM_REGION:NUM_REGION * 2]

    @property
    def G(self):
        return self.param[NUM_REGION * 2]

    @property
    def sigma(self):
        return self.param[NUM_REGION * 2 + 1:]

    @property
    def FC_CORR_cost(self):
        return self.performance[0]

    @property
    def FC_L1_cost(self):
        return self.performance[1]

    @property
    def FCD_KS(self):
        return self.performance[2]

    @property
    def total_cost(self):
        return self.performance[3]

    @property
    def pyg_nodes(self):
        all_node_feats = []
        for i in range(0, NUM_REGION):
            node_feature = torch.tensor([self.wEE[i], self.wEI[i], self.sigma[i]], device=self.device)
            all_node_feats.append(node_feature)
        x = torch.stack(all_node_feats, dim=0)
        return x

    def to_numpy(self):
        param_with_performance = torch.cat((self.performance, self.param))
        return param_with_performance.cpu().numpy()


class ParamCoefCost(ParamPerformance):

    @property
    def G(self):
        return self.param[-1]
