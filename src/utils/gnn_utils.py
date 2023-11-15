import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from src.basic.cost_type import CostType
from src.basic.constants import NUM_REGION


def edge_from_SC(SC: torch.Tensor):
    assert SC.shape == (NUM_REGION, NUM_REGION)
    sparse_SC = SC.to_sparse()
    edge_index = sparse_SC.indices().contiguous()
    edge_weight = sparse_SC.values()

    return edge_index, edge_weight


class RebatchNodeFeat(torch.nn.Module):
    """
    Re-batch node features in a batch as graphs in a batch in pyg are merged into a large graph during training
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: merged node features in a batch
            num_graphs: number of graphs within one batch
        """
        batch_size = x.shape[0] // NUM_REGION
        x = x.view(batch_size, -1)
        # x now has shape (batch_size, NUM_REGION * num_node_features)
        return x
