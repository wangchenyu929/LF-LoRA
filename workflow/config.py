import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser
from datetime import datetime, timedelta


# ===== Define and parse arguments =====
@dataclass
class Arguments:
    # LLM arguments
    base_model_path: Optional[str] = field(
        # default="/root/nfs/LLMDatasetsAndModels/models/TinyLlama-1.1B-intermediate-step-1431k-3T",
        # default="/root/nfs/LLMDatasetsAndModels/models/bert-base-cased",
        # default="/dataset/aigc_modelzoo/models/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
        default="/dataset/aigc_modelzoo/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
        metadata={"help": "the path of base model"},
    )
    model_name: Optional[str] = field(
        # default="TinyLlama-1.1B-intermediate-step-1431k-3T",
        # default="bert-base-cased",
        default="Llama-3.2-1B-Instruct",
        # default="Llama-3.2-3B-Instruct",
        metadata={"help": "the name of base model"},
    )
    dataset_path: Optional[str] = field(
        # default="/root/nfs/LLMDatasetsAndModels/datasets/medalpaca/medical_meadow_medical_flashcards",
        default="/root/nfs/LLMDatasetsAndModels/datasets/FinGPT/fingpt-sentiment-train",
        # default="/root/nfs/LLMDatasetsAndModels/datasets/yelp_review_full",
        # default="/root/nfs/LLMDatasetsAndModels/datasets/vicgalle/alpaca-gpt4",
        # default="/root/nfs/LLMDatasetsAndModels/datasets/wikitext/wikitext-2-raw-v1",
        metadata={"help": "the path of dataset"},
    )
    dataset_name: Optional[str] = field(
        default="fingpt-sentiment-train",
        # default="alpaca-gpt4",
        # default="yelp_review_full",
        # default="medical_meadow_medical_flashcards",
        metadata={"help": "the name of dataset"},
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    k_group_clients: Optional[List[int]] = field(
        default_factory=lambda: [0, 1, 2, 3],
        metadata={
            "help": "The client group used to train the key matrix. The first one is the key edge server"
        },
    )
    q_group_clients: Optional[List[int]] = field(
        default_factory=lambda: [4, 5, 6, 7],
        metadata={
            "help": "The client group used to train the query matrix. The first one is the query edge server"
        },
    )
    v_group_clients: Optional[List[int]] = field(
        default_factory=lambda: [8, 9, 10, 11],
        metadata={
            "help": "The client group used to train the value matrix. The first one is the value edge server"
        },
    )
    o_group_clients: Optional[List[int]] = field(
        default_factory=lambda: [12, 13, 14, 15],
        metadata={
            "help": "The client group used to train the output matrix. The first one is the output edge server"
        },
    )
    freeze_layers_num: Optional[int] = field(
        default=4, metadata={"help": "The number of freeze layers"}
    )

    # FL arguments
    num_rounds: Optional[int] = field(
        default=30, metadata={"help": "the number of global aggregation rounds"}
    )
    num_clients: Optional[int] = field(
        default=16, metadata={"help": "the number of clients"}
    )
    edge_clients: Optional[List[int]] = field(
        default_factory=lambda: [0, 4, 8, 12],
        metadata={"help": "the number of edge aggregation rounds"},
    )
    edge_aggregate_round: Optional[int] = field(
        default=1, metadata={"help": "the number of edge aggregation rounds"}
    )
    split_strategy: Optional[str] = field(
        default="quality-non-iid",
        metadata={
            "help": "the split strategy:assign, non-iid, quality-non-iid and iid"
        },
    )
    data_sample: Optional[int] = field(
        default=3000, metadata={"help": "the data sample number of total client"}
    )
    total_data_sample: Optional[int] = field(
        default=76500, metadata={"help": "the data sample number of total client"}
    )
    dirichlet_alpha: Optional[float] = field(
        default=1,
        metadata={
            "help": "the dirichlet alpha for non-iid and quality-non-iid data distribution"
        },
    )
    data_distribution: Optional[List[int]] = field(
        default_factory=lambda: [
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
        ],
        metadata={
            "help": "The client group used to train the query matrix. The first one is the query edge server"
        },
    )
    seed: Optional[int] = field(default=2024, metadata={"help": "the seed to use"})


# ===== get arguments config=====
def get_args_config():
    """
    Parses command-line arguments into a dataclass instance.

    Returns:
        config_args: An object containing all parsed configuration arguments.
    """
    parser = HfArgumentParser((Arguments,))
    (config_args,) = parser.parse_args_into_dataclasses()
    return config_args


def save_config(training_args):
    """
    Saves the training configuration to a JSON file in a uniquely created output directory.

    Args:
        training_args: An object containing all training-related configuration parameters.
    """
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = training_args.dataset_name
    model_name_split = training_args.model_name
    output_dir = f"{training_args.output_dir}/{dataset_name_split}_{model_name_split}_{training_args.num_rounds}_{now_time}"
    while True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{training_args.output_dir}/{dataset_name_split}_{model_name_split}_{training_args.num_rounds}_{now_time}"
    training_args.output_dir = output_dir

    # convert arguments to json and save it
    with open(os.path.join(training_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "training_args": asdict(training_args),
        }
        json.dump(combined_dict, f, indent=4)
