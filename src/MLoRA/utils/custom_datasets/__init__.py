"""
    High level functions for model training
"""
from typing import Optional

import numpy as np
# from .extra_rm_datasets import load_anthropic_rlhf, load_hellaswag, load_shp
# from .instruction import INSTRUCTION_DATASETS, InstructionDataset
# from .oasst_dataset import load_oasst_export
# from .prompt_dialogue import Gpt4All, load_oig_file
# from .qa_datasets import (
#     SODA,
#     DatabricksDolly15k,
#     JokeExplaination,
#     QADataset,
#     SODADialogue,
#     TranslatedQA,
#     Vicuna,
#     WebGPT,
#     load_alpaca_dataset,
# )
# from model_training.custom_datasets.MyDialogue import MyDialogue
from .My_custom import MyCustom,My_custom_data
# from model_training.custom_datasets.rank_datasets import AugmentedOA
# from model_training.custom_datasets.summarization import HFSummary, HFSummaryPairs, SummarizationDataset
# from model_training.custom_datasets.toxic_conversation import ProsocialDialogue, ProsocialDialogueExplaination
# from model_training.custom_datasets.translation import WMT2019, DiveMT, TEDTalk
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def train_val_dataset(dataset, val_split=0.2) -> tuple[Dataset, Dataset | None]:
    if val_split == 0:
        return dataset, None

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=666, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_one_dataset(
    conf,
    dataset_name: str,
    val_split: float = 0.2,
    data_path: str = None,
    mode: str = "sft",
    max_val_set: Optional[int] = None,
    ssize=None,
    **kwargs,
) -> tuple[Dataset, Dataset | None]:
    # print(dataset_name,dataset_name in MY_INSTRUCTION_DATASETS,dataset_name in My_Dialogues)
    if mode == "rl":
        assert dataset_name in RL_DATASETS, f"Dataset {dataset_name} not supported for RL"

    if mode == "rm":
        assert dataset_name in RM_DATASETS, f"Dataset {dataset_name} not supported for reward modeling"

    data_path = data_path or conf["cache_dir"]
    dataset_name = dataset_name.lower()
    if dataset_name in My_custom_data:
        dataset = MyCustom(mode=mode, cache_dir=My_custom_data[dataset_name])
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # if eval not already defined
    if not ("eval" in locals() and "train" in locals()):
        train, eval = train_val_dataset(dataset, val_split=val_split)
    # print(**kwargs)
    if eval and max_val_set and len(eval) > max_val_set:
        subset_indices = np.random.choice(len(eval), max_val_set)
        eval = Subset(eval, subset_indices)
    if ssize and len(train) > ssize:
        subset_indices = np.random.choice(len(train), ssize)
        train = Subset(train, subset_indices)

    return train, eval
