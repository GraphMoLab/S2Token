import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader.dataloader import Collater

from pe import pad_coarsen_adj
from pipeline.data_utils.constants import IGNORE_INDEX


def eval_func(text_batch):
    new_batch = []
    for t in text_batch:
        label_mask = t["labels"] != IGNORE_INDEX
        input_mask = t["labels"] == IGNORE_INDEX
        input_ids = t["input_ids"][input_mask]
        attention_mask = t["attention_mask"][input_mask]
        labels = t["labels"][label_mask]
        seq_length = input_ids.size(0)
        new_batch.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seq_length": seq_length,
        })
    return new_batch


def batchify(batch, tokenizer, max_length: int, use_trunc=True, is_eval=False):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    # 过滤掉None的样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:  # 防止整个batch被过滤
        return None

    output_batch = {}
    graph_list = [data["graph"] for data in batch]
    smiles_list = [data["smiles"] for data in batch]
    coarsen_adj_list = [data["coarsen_adj"] for data in batch]

    num_graphs_per_sample = torch.LongTensor([graph is not None for graph in graph_list])
    # 1. remove None images from graph_list
    graph_list = [graph for graph in graph_list if graph is not None]

    # -------------------------------------------
    motif_sizes = [
        graph.num_substructs + 1 if graph.num_substructs == 0
        else graph.num_substructs
        for graph in graph_list
    ]
    batch_max_motif_size = max(motif_sizes)

    for item, g in zip(batch, graph_list):
        item['text_input'] = tokenizer.encode_prompt(
            item['raw_text'],
            g,
            max_length,
            batch_max_motif_size,
        )

    # 2. collate for images: [num_images, c, h, w]
    if len(graph_list) == 1:
        # print('单个图的graph_list')
        output_batch["graph_values"] = graph_list[0]
    else:
        try:
            output_batch["graph_values"] = Collater([], [])(graph_list)
        except:
            print(graph_list)
    # 3. collate for text
    text_batch = [item["text_input"] for item in batch]
    if is_eval:
        text_batch = eval_func(text_batch)
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    # text_batch.pop("attention_mask")

    output_batch.update({
        **text_batch,
        "num_graphs": num_graphs_per_sample,
    })

    # output_batch = {k: v.to("cuda") for k, v in output_batch.items()}
    output_batch.update({"smiles": smiles_list})

    substruct_indexer_list = [g["substruct_indexer"] for g in graph_list]
    substruct_nodes_list = [g["substruct_nodes"] for g in graph_list]
    coarsen_adj = torch.stack([pad_coarsen_adj(coarsen, batch_max_motif_size) for coarsen in coarsen_adj_list])
    structure_pes = pad_sequence([g["substructure_pe"] for g in graph_list], batch_first=True, padding_value=0)

    output_batch.update({
        "substruct_indexer": substruct_indexer_list,
        "substruct_nodes": substruct_nodes_list,
        "coarsen_adj": coarsen_adj,
        "structure_pes": structure_pes
    })

    return output_batch


