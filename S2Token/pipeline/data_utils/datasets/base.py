import random
from dataclasses import dataclass

from PIL import Image
# from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset

from pe import RWSE
from pipeline.data_utils.constants import SYSTEM_MESSAGE, HUMAN, AI, MEDIA_TOKENS
from typing import List

from rdkit import Chem

import torch
import utils
from preprocess_mol_data import Substructure_transform


class BaseDataset(InMemoryDataset):
    """Base dataset class
    """

    def __init__(self,
                 input_path,
                 tokenizer,
                 max_length,
                 **kwargs):
        super(BaseDataset, self).__init__()
        self.data, self.slices = torch.load(input_path)

        self.input_path = input_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = None

        self.smiles_prompt = kwargs.pop("smiles_prompt", None)
        self.cluster_shuffle = kwargs.pop("cluster_shuffle", False)

        # self.valid_indices = self._precompute_valid_indices()

        # TODO: need to check utils
        if kwargs and utils.is_main_process():
            print("=" * 80)
            print("Dataset ignore kwargs: {}".format(kwargs))
            print("=" * 80)

    def preprocess_data(self, data):
        """ perform pre-processing for the given data if required
        Args:
            data: datapoint given from self.data
        """
        return data

    def build_text_from_data(self, data):
        """Build instruction text from data
        Args:
            data: datapoint given from self.data
        """
        temp_text = list()

        instruction = data.instruction

        mol = Chem.MolFromSmiles(data.smiles)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        if not self.smiles_prompt:
            smiles_input = canonical_smiles
        else:
            smiles_input = self.smiles_prompt.format(canonical_smiles)

        graph_input = MEDIA_TOKENS["graph"][0]
        temp_text.append({"role": "system", "content": SYSTEM_MESSAGE})

        if isinstance(instruction, list):
            # limit the length of smiles_input to 128
            if graph_input not in instruction[0]['content']:
                instruction[0]['content'] = smiles_input[:128] + graph_input + " " + instruction[0]['content']
                # instruction[0]['content'] = smiles_input + graph_input + " " + instruction[0]['content']
            else:
                instruction[0]['content'] = smiles_input[:128] + " " + instruction[0]['content']
                # instruction[0]['content'] = smiles_input + " " + instruction[0]['content']

            temp_text.extend(instruction)
            temp_text.append({"role": "assistant", "content": data.output.strip()})

        elif isinstance(instruction, str):
            if graph_input not in instruction:  # 'COc1cc(C=CC(=O)OCC2OC(OCC3=CCC4C(CO)C(=O)OCC34)C(O)C(O)C2O)cc(OC)c1O<graph> Provide a brief overview of this molecule.'
                aug_instruction = smiles_input[:128] + graph_input + " " + instruction
                # aug_instruction = smiles_input + graph_input + " " + instruction
            else:
                aug_instruction = smiles_input[:128] + " " + instruction
                # aug_instruction = smiles_input + " " + instruction
            temp_text.append({"role": "user", "content": aug_instruction})
            temp_text.append({"role": "assistant", "content": data.output.strip()})
        else:
            raise ValueError("Instruction should be either list or string")

        return temp_text

    def process_data(self, data):
        """Process data for the given data if required
        Args:
            data: datapoint given from self.data
        """
        data = self.preprocess_data(data)

        text = self.build_text_from_data(data)
        # text_input = self.tokenizer.encode_prompt(
        #     text,
        #     self.max_length
        # )

        mol = Chem.MolFromSmiles(data.smiles)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

        # --------------------------------------------------
        data.mol = mol
        data = Substructure_transform(data)
        # print(data.fragment_types)

        # 初始化 node_mask
        node_mask = torch.zeros(data.num_substructs, data.num_nodes, dtype=torch.bool)

        # 填充 node_mask
        for i in range(data.num_substructs):
            start = data.substruct_ptr[i]
            end = data.substruct_ptr[i + 1]
            nodes_in_patch = data.substruct_nodes[start:end]
            node_mask[i, nodes_in_patch] = True

        # 计算 coarsen_adj
        mask = node_mask.float()
        coarsen_adj = torch.matmul(mask, mask.t())

        # print("node_mask shape:", node_mask.shape)  # (10, num_nodes)
        # print("coarsen_adj shape:", coarsen_adj.shape)  # (10, 10)

        # print(data.higher_edge_index)
        #
        # print(data.substruct_id)
        # print(data.substruct_nodes)
        # print(data.substruct_ptr)
        # print(data.num_substructs)

        return {
            "graph": data,
            "raw_text": text,
            # "text": text_input,
            "smiles": canonical_smiles,
            "coarsen_adj": coarsen_adj,
        }

    def __getitem__(self, index):
        while True:
            data = self.get(index)
            data = self.process_data(data)
            if data['graph'].num_substructs <= 128 and data['graph'].num_substructs !=0:
                return data
            index = random.randint(0, len(self) - 1)

        # while True:
        #     data = self.get(index)
        #     data = self.process_data(data)
        #     if data['graph'].num_substructs != 0:
        #         return data
        #     index = random.randint(0, len(self) - 1)

        # data = self.get(index)
        # data = self.process_data(data)
        # return data

