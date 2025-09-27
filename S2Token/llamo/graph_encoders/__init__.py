import torch
from torch import nn

from .gMHA_hadamard import HadamardEncoderLayer, FragEncoder
from .molecule_stm import GNNstm, GNN_graphpred


def build_encoder(config):
    JK = config.JK if "JK" in config.keys() else "last"
    cat_grep = config.cat_grep if "cat_grep" in config.keys() else True
    molecule_node_model = GNNstm(
        num_layer=config.gin_num_layers,
        emb_dim=config.hidden_size,
        gnn_type='gin',
        drop_ratio=config.gin_drop_ratio,
        JK=JK,
    )

    graph_encoder = GNN_graphpred(
        emb_dim=config.hidden_size,
        molecule_node_model=molecule_node_model,
        init_checkpoint="MoleculeSTM/molecule_model.pth",
        cat_grep=cat_grep
    )

    ln_graph = LayerNorm(4096)

    project_pe = nn.ModuleList([
        nn.Sequential(
            nn.Linear(16, 512)
        ) for _ in range(6)
    ])

    project_node_outputs = nn.ModuleList([
        nn.Sequential(
            nn.Linear(300, 512)
        ) for _ in range(6)
    ])

    project_structure_pe = nn.ModuleList([
        nn.Sequential(
            nn.Linear(8, 512)
        ) for _ in range(6)
    ])

    transformer_encoder = nn.ModuleList([
        HadamardEncoderLayer(
                d_model=512,
                dim_feedforward=1024,
                nhead=8,
                batch_first=True,
                dropout=0.1
            ) for _ in range(4)
        ])

    return graph_encoder, ln_graph, project_pe, project_structure_pe, project_node_outputs, transformer_encoder


class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
