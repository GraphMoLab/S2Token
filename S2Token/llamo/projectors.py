from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
# from timm.layers import LayerNorm, LayerNorm2d
from torch.nn import LayerNorm

from .configuration_llamo import GraphProjectorConfig
from .graph_encoders import LayerNorm

from llamo.chem import mol_to_graphs
from rdkit import Chem


def build_pos_embeds(
        config: GraphProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: GraphProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: GraphProjectorConfig):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

class Pooling(nn.Module):
    def __init__(
            self, hidden_size: int, num_queries: int = 32, mode: str = "attn"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.mode = mode

        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, mask=None):
        y = self.linear(x) / (self.linear.weight.norm() + 1e-6)
        if mask is not None:
            added_value = (~mask) * (-1000000)
            y += added_value.unsqueeze(-1)
        topk_values, topk_indices = y.topk(self.num_queries, dim=1)
        x = torch.gather(x, 1, topk_indices.repeat(1, 1, x.size(-1))) * topk_values.tanh()
        return x


class MultilevelProjector(nn.Module):
    """Base projector class"""

    def __init__(
            self,
            config: GraphProjectorConfig,
            num_query_tokens: int,
            output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.num_query_tokens = num_query_tokens

        self.cat_num = 0

        self.eos_tokens = None
        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        query_tokens = self._forward(x)  # (B, L, output_hidden_size)

        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class MLPMultilevelProjector(MultilevelProjector):
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, 512 * 6, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x

