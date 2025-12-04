import copy
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)
from accelerate import Accelerator
from peft import get_peft_model, get_peft_model_state_dict, LoraConfig


def init_global_model(training_configs, train_dataset, model_name):
    """
    Initialize the global model with LoRA configuration.

    Args:
        training_configs: Configuration object with model paths and parameters.
        train_dataset: The dataset used to determine label information (for classification models).
        model_name (str): Name of the model architecture.

    Returns:
        model: The initialized global model with LoRA adaptation applied.
        init_global_model_dict: Deep-copied initial LoRA state dict.
    """
    if model_name in ["bert-base-cased"]:
        # ===== for bert-base-cased and yelp-review-full start=====
        num_labels = train_dataset.features["label"].num_classes
        class_names = train_dataset.features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")
        # Create an id2label mapping, we will need this for our classifier
        id2label = {i: label for i, label in enumerate(class_names)}

        model = AutoModelForSequenceClassification.from_pretrained(
            training_configs.base_model_path, id2label=id2label
        )

        # the initial global model config
        LConfig = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, LConfig)
        # ===== for bert-base-cased and yelp-review-full end =====
    elif model_name in [
        "Llama-2-7b-hf",
        "TinyLlama-1.1B-intermediate-step-1431k-3T",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
    ]:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {
            "": Accelerator().local_process_index
        }  # Copy the model to each device
        torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            training_configs.base_model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        LConfig = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, LConfig)
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    init_global_model_dict = copy.deepcopy(get_peft_model_state_dict(model))
    return model, init_global_model_dict


# ==== split aggregate lora to k q v =====
def split_layers(complete_lora, model_name):
    """
    Split the LoRA state dict into separate matrices based on layer types.

    Args:
        complete_lora (dict): Full LoRA model state dictionary.
        model_name (str): Model name to determine layer naming conventions.

    Returns:
        Tuple of dicts: (k_lora, q_lora, v_lora) or (k_lora, q_lora, v_lora, o_lora)
    """
    k_lora = {}
    q_lora = {}
    v_lora = {}
    if model_name in ["bert-base-cased"]:
        for key, value in complete_lora.items():
            if "key" in key:
                k_lora[key] = value
            elif "query" in key:
                q_lora[key] = value
            elif "value" in key:
                v_lora[key] = value
            elif "classifier" in key:
                k_lora[key] = value
                q_lora[key] = value
                v_lora[key] = value
        return k_lora, q_lora, v_lora
    elif model_name in [
        "Llama-2-7b-hf",
        "TinyLlama-1.1B-intermediate-step-1431k-3T",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
    ]:
        o_lora = {}
        for key, value in complete_lora.items():
            if "k_proj" in key:
                k_lora[key] = value
            elif "q_proj" in key:
                q_lora[key] = value
            elif "v_proj" in key:
                v_lora[key] = value
            elif "o_proj" in key:
                o_lora[key] = value
            else:
                k_lora[key] = value
                q_lora[key] = value
                v_lora[key] = value
                o_lora[key] = value
        return k_lora, q_lora, v_lora, o_lora


def global_aggregate_traditional(
    global_dict, local_dict_list, sample_num_list, clients_this_round
):
    """
    Perform weighted averaging of local model weights for global aggregation.

    Args:
        global_dict (dict): Initial global model weights.
        local_dict_list (List[dict]): List of local model weight dicts from clients.
        sample_num_list (List[int]): Number of samples per client.
        clients_this_round (List[int]): Indices of participating clients.

    Returns:
        dict: Aggregated global model dictionary.
    """
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    for key in global_dict.keys():
        global_dict[key] = sum(
            [
                local_dict_list[client][key]
                * sample_num_list[client]
                / sample_this_round
                for client in clients_this_round
            ]
        )
    return global_dict


def global_aggregate_3matrix(k_lora, q_lora, v_lora):
    """
    Aggregate LoRA weights from key/query/value matrices (used in BERT).

    Args:
        k_lora, q_lora, v_lora (dict): LoRA weights for each matrix.

    Returns:
        dict: Aggregated LoRA weights.
    """
    # Make sure all tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_lora = {}

    for layer in k_lora.keys():
        if "attention.self" in layer:
            for component in ["query", "key", "value"]:
                for weight_type in ["lora_A.weight", "lora_B.weight"]:
                    new_key = layer.replace("key", component)
                    if component == "key":
                        aggregated_lora[new_key] = k_lora[new_key].to(device)
                    elif component == "query":
                        aggregated_lora[new_key] = q_lora[new_key].to(device)
                    elif component == "value":
                        aggregated_lora[new_key] = v_lora[new_key].to(device)
        elif "classifier" in layer:
            aggregated_lora[layer] = (
                k_lora[layer].to(device)
                + q_lora[layer].to(device)
                + v_lora[layer].to(device)
            ) / 3
            # aggregated_lora[layer] = q_lora[layer].to(device)

    return aggregated_lora


def global_aggregate_4matrix(k_lora, q_lora, v_lora, o_lora):
    """
    Aggregate LoRA weights from key/query/value/output matrices (used in LLaMA).

    Args:
        k_lora, q_lora, v_lora, o_lora (dict): LoRA weights for each matrix.

    Returns:
        dict: Aggregated LoRA weights.
    """
    # Make sure all tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_lora = {}

    for layer in k_lora.keys():
        if "self_attn" in layer:
            for component in ["k_proj", "q_proj", "v_proj", "o_proj"]:
                for weight_type in ["lora_A.weight", "lora_B.weight"]:
                    new_key = layer.replace("k_proj", component)
                    if component == "k_proj":
                        aggregated_lora[new_key] = k_lora[new_key].to(device)
                    elif component == "q_proj":
                        aggregated_lora[new_key] = q_lora[new_key].to(device)
                    elif component == "v_proj":
                        aggregated_lora[new_key] = v_lora[new_key].to(device)
                    elif component == "o_proj":
                        aggregated_lora[new_key] = o_lora[new_key].to(device)

        else:
            aggregated_lora[layer] = (
                k_lora[layer].to(device)
                + q_lora[layer].to(device)
                + v_lora[layer].to(device)
                + o_lora[layer].to(device)
            ) / 4

    return aggregated_lora


def edge_aggregate_layer(save_dir, group_clients):
    """
    Aggregate LoRA weights from a group of clients at the edge (e.g. edge server).

    Args:
        save_dir (str): Directory where LoRA checkpoints are stored.
        group_clients (List[int]): List of client IDs in the group.

    Returns:
        dict: Averaged LoRA weights across the group.
    """
    # Make sure all tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_lora = {}
    # Count the number of occurrences and the total number of occurrences of each layer
    layer_sums = {}
    layer_counts = {}

    for client in group_clients:
        lora_model_path = os.path.join(save_dir, f"lora_checkpoint_client_{client}.pt")
        lora_state_dict = torch.load(lora_model_path)
        for layer, weights in lora_state_dict.items():
            weights = weights.to(device)
            if layer not in layer_sums:
                layer_sums[layer] = weights.clone()
                layer_counts[layer] = 1
            else:
                layer_sums[layer] += weights
                layer_counts[layer] += 1
    # calculate the average
    for layer in layer_sums:
        aggregated_lora[layer] = layer_sums[layer] / layer_counts[layer]

    return aggregated_lora
