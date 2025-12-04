# spliting the datasets to each client
import datasets
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from config import get_args_config

training_configs = get_args_config()
dataset_name = training_configs.dataset_name


def preprocess_dateset():
    """
    Preprocesses the dataset based on the dataset name and split strategy.

    Returns:
        train_shard_dataset: Tokenized and shuffled training dataset.
        eval_shard_dataset: Tokenized and optionally sharded evaluation dataset.
    """

    # decide the total_data_sample
    if training_configs.split_strategy == "iid":
        total_data_sample = training_configs.data_sample * training_configs.num_clients
    elif (
        training_configs.split_strategy == "quality-non-iid"
        or training_configs.split_strategy == "non-iid"
    ):
        total_data_sample = training_configs.total_data_sample
    elif training_configs.split_strategy == "assign":
        total_data_sample = sum(training_configs.data_distribution)
    else:
        total_data_sample = 100000

    # preprocess dataset
    if dataset_name in ["yelp_review_full", "20_newsgroups"]:
        dataset = load_dataset(training_configs.dataset_path)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        base_model = training_configs.base_model_path
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length")

        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
        train_shard_dataset = tokenized_train_dataset.shuffle(
            training_configs.seed
        ).select(range(total_data_sample))
        # one shard (0) is used for evaluation and the other shard (1) is used for test
        eval_shard_dataset = tokenized_eval_dataset.shard(num_shards=2, index=0).select(
            range(500)
        )

    elif dataset_name in [
        "fingpt-sentiment-train",
        "alpaca-gpt4",
        "medical_meadow_medical_flashcards",
    ]:
        dataset = load_dataset(training_configs.dataset_path, split="train")
        processes_dataset = process_sft_dataset(
            dataset_name, dataset, training_configs.total_data_sample
        )
        eval_shard_dataset = processes_dataset.shuffle(training_configs.seed).select(
            range(
                training_configs.total_data_sample - 500,
                training_configs.total_data_sample,
            )
        )  # the last 500 sample serve as evaluate dataset
        # split_dataset = processes_dataset.train_test_split(test_size=0.1, seed=42)
        train_shard_dataset = processes_dataset.shuffle(training_configs.seed).select(
            range(total_data_sample)
        )
        # train_shard_dataset = split_dataset['train']
        # eval_shard_dataset = split_dataset['test']

    return train_shard_dataset, eval_shard_dataset


def split_dataset(train_dataset, eval_dataset):
    """
    Splits the training and eval dataset among clients according to the split strategy.

    Returns:
        local_train_datasets: List of training datasets for each client.
        local_eval_datasets: List of evaluation datasets for each client.
    """
    seed = training_configs.seed
    train_dataset = train_dataset.shuffle(seed)  # Shuffle the dataset
    eval_dataset = eval_dataset.shuffle(seed)
    local_train_datasets = []
    local_eval_datasets = []
    if training_configs.split_strategy == "iid":
        for i in range(training_configs.num_clients):
            local_train_datasets.append(
                train_dataset.shard(training_configs.num_clients, i)
            )
            local_eval_datasets.append(
                eval_dataset.shard(training_configs.num_clients, i)
            )
    elif training_configs.split_strategy == "quality-non-iid":
        total_data_sample = training_configs.total_data_sample
        dataset_shards = training_configs.num_clients + 1
        # Generate proportions using Dirichlet distribution
        np.random.seed(training_configs.seed)
        while True:
            # Generate proportions using Dirichlet distribution
            proportions = np.random.dirichlet(
                alpha=[training_configs.dirichlet_alpha] * dataset_shards
            )

            # Calculate sizes based on proportions
            sizes = (proportions * total_data_sample).astype(int)

            # Ensure each client gets at least min_samples_per_client
            remaining = total_data_sample - sizes.sum()
            for i in range(dataset_shards):
                if sizes[i] < 10:
                    to_add = 10 - sizes[i]
                    sizes[i] += to_add
                    remaining -= to_add

            # Distribute any remaining samples
            if remaining > 0:
                for i in range(remaining):
                    sizes[i % dataset_shards] += 1

            # Check if all clients have at least min_samples_per_client
            if np.all(sizes >= dataset_shards):
                break

        client_data_sizes = sizes.tolist()

        # assign dataset
        start_index = 0
        for i, size in enumerate(client_data_sizes):
            end_index = start_index + size
            local_train_datasets.append(
                train_dataset.select(range(start_index, end_index))
            )
            # local_eval_datasets.append(eval_dataset.skip(start_index).take(size))
            local_eval_datasets.append(
                eval_dataset.shard(
                    training_configs.num_clients + 1, i
                )  # the last one is global server
            )
            start_index = end_index

    elif training_configs.split_strategy == "assign":
        # client_data_sizes = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 5000, 5000, 5000, 5000]
        client_data_sizes = training_configs.data_distribution
        start_index = 0
        for i, size in enumerate(client_data_sizes):
            end_index = start_index + size
            local_train_datasets.append(
                train_dataset.select(range(start_index, end_index))
            )
            # local_eval_datasets.append(eval_dataset.skip(start_index).take(size))
            local_eval_datasets.append(
                eval_dataset.shard(
                    training_configs.num_clients + 1, i
                )  # the last one is global server
            )
            start_index = end_index

    return local_train_datasets, local_eval_datasets


def process_sft_dataset(dataset_name, dataset, dataset_sample):
    """
    Processes SFT (supervised fine-tuning) datasets into a unified format.

    Args:
        dataset_name (str): Name of the dataset.
        dataset: Raw dataset loaded from Huggingface.
        dataset_sample (int): Max number of samples to use.

    Returns:
        dataset: Processed and optionally trimmed dataset.
    """
    # if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
    if dataset_name in [
        "CodeAlpaca-20k",
        "alpaca-cleaned",
        "fingpt-sentiment-train",
        "medical_meadow_medical_flashcards",
    ]:
        dataset = dataset.map(
            alpaca_format,
            remove_columns=["input", "output"],
            desc=f"Preprocessing {dataset_name} for unified format.",
        )
    # elif dataset_name in [WizardLM/WizardLM_evol_instruct_70k"]:
    elif dataset_name in ["WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    # elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
    elif dataset_name in ["alpaca", "alpaca-gpt4", "finance-alpaca"]:
        dataset = dataset.map(
            alpaca_format,
            remove_columns=["input", "output", "text"],
            desc=f"Preprocessing {dataset_name} for unified format.",
        )
    # elif dataset_name in ["TIGER-Lab/MathInstruct"]:
    elif dataset_name in ["MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=["instruction"])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(["source"])
    # elif dataset_name in ["lighteval/MATH"]:
    elif dataset_name in ["MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(["level", "type"])
    elif dataset_name in ["gsm8k"]:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    # elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:
    elif dataset_name in [
        "medical_meadow_medical_flashcards"
    ]:  # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(
        f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. ====="
    )
    return dataset


def alpaca_format(example):
    """
    Formats a single example into Alpaca-style instruction-response pairs.

    Args:
        example (dict): A sample containing 'instruction', 'input', and 'output'.

    Returns:
        dict: Reformatted example with combined instruction and new 'response' field.
    """
    if example["input"] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example["input"]
    example["response"] = example["output"]
    return example
