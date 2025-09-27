import json

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, add_positional_encoding

import fragmentations as frag
from torch_geometric.data import InMemoryDataset, Batch
from tqdm import tqdm

from pe import RWSE

conversations = {
    'id': 2,
    'motif': [2, 471, 565, -500],
    'conversations': [
        {'from': 'human',
         'value': 'Given a molecular representation: <graph>, provide a description of this molecule.'},
        {'from': 'gpt',
         'value': 'Reinforcement_Learning'}
    ],
}


def generate_conversation(input_data):
    conversations["id"] = input_data["id"]
    conversations["motif"] = input_data["motif"]
    conversations["conversations"][0]["value"] = input_data["conversations"][0]["value"]
    conversations["conversations"][1]["value"] = input_data["conversations"][1]["value"]
    return json.dumps(conversations, ensure_ascii=False)


class Graph2Subsequence(InMemoryDataset):
    def __init__(self, root='datasets', transform=None, pre_transform=None):
        self.original_root = root
        self.folder = 'S2Token/stage1'

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_structure_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        geometric_data_processed_gpse = torch.load(
            'MoleculeDesc/train.pt')

        data_list = []
        if self.pre_transform is not None:
            for idx, g in tqdm(enumerate(geometric_data_processed_gpse), total=len(geometric_data_processed_gpse)):
                mol = Chem.MolFromSmiles(g.smiles)

                g.mol = mol
                g_pre_transform = self.pre_transform(g)
                data_list.append(g_pre_transform)

        print('Saving...')
        torch.save(data_list, self.processed_paths[0])


def Substructure_transform(data):
    fragmentation_method = ["RingsPaths",
                            "higher_level_graph_tree",
                            {"vocab_size": 30, "max_ring": 15}
                            ]
    higher_edge_features = False

    frag_type, frag_usage, vocab_size_params = fragmentation_method

    frag_constructions = {
        "BRICS": frag.BRICS,
        "Rings": frag.Rings,
        "RingsEdges": frag.RingsEdges,
        "RingsPaths": frag.RingsPaths,
    }
    frag_usages = {
        "node_feature":
            frag.NodeFeature(vocab_size_params["vocab_size"]),
        "global":
            frag.GraphLevelFeature(vocab_size_params["vocab_size"]),
        "fragment":
            frag.FragmentRepresentation(vocab_size_params["vocab_size"]),
        "higher_level_graph_tree":
            frag.HigherLevelGraph(vocab_size_params["vocab_size"],
                                  "tree",
                                  higher_edge_features=higher_edge_features),
        "higher_level_graph_node":
            frag.HigherLevelGraph(vocab_size_params["vocab_size"],
                                  "node",
                                  higher_edge_features=higher_edge_features)
    }

    transformations = []
    random_walk_encoding = add_positional_encoding.AddRandomWalkPE(16)
    transformations.append(random_walk_encoding)
    frag_construction = frag_constructions[frag_type](**vocab_size_params)
    frag_representation = frag_usages[frag_usage]
    transformations.append(frag_construction)
    transformations.append(frag_representation)

    transform = Compose(transformations)
    data = transform(data)

    if data.num_substructs != 0:
        substruct_ptr = data.substruct_ptr.tolist()

        # Calculate the number of nodes in each substructure using list comprehension
        substruct_contain_node_num = [
            substruct_ptr[idx + 1] - substruct_ptr[idx]
            for idx in range(data.num_substructs)
        ]

        # Create the indexer tensor
        indexer = torch.repeat_interleave(
            torch.arange(data.num_substructs),
            torch.tensor(substruct_contain_node_num)
        )
        data.substruct_indexer = indexer
    else:
        data.substruct_indexer = torch.tensor([0] * data.num_nodes)
        data.substruct_nodes = torch.arange(data.num_nodes)

    substructure_pe = RWSE(data.higher_edge_index, 8, data.num_substructs)
    data.substructure_pe = substructure_pe[data.substruct_id]

    return data


if __name__ == '__main__':
    class BaseDataset(InMemoryDataset):

        def __init__(self,
                     **kwargs):
            super(BaseDataset, self).__init__()
            self.data = torch.load('chebi/test/test.pt')

        def __getitem__(self, index):
            data = self.get(index)
            mol = Chem.MolFromSmiles(data.smiles)
            data.mol = mol
            data = Substructure_transform(data)

            return data

    dataset = BaseDataset()
