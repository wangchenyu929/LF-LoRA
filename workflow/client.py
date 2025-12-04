import os
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)
from accelerate import Accelerator
from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict,
    get_peft_model,
    LoraConfig,
)
from config import get_args_config


def generate_lora_config(client_id: int):
    """
    Generate a LoRA configuration based on the client ID.

    Args:
        client_id (int): The ID of the client.

    Returns:
        LoraConfig: A LoRA configuration customized for the client.
    """
    training_configs = get_args_config()
    model_name = training_configs.model_name

    # Determine target_modules through training_configs.
    if model_name in ["bert-base-cased"]:
        target_modules_mapping = {
            "k_group_clients": "key",
            "q_group_clients": "query",
            "v_group_clients": "value",
        }
    elif model_name in [
        "Llama-2-7b-hf",
        "TinyLlama-1.1B-intermediate-step-1431k-3T",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
    ]:
        target_modules_mapping = {
            "k_group_clients": "k_proj",
            "q_group_clients": "q_proj",
            "v_group_clients": "v_proj",
            "o_group_clients": "o_proj",
        }

    target_module = None
    for group_name, target in target_modules_mapping.items():
        if client_id in getattr(training_configs, group_name):
            target_module = target
            break

    if target_module is None:
        raise ValueError(
            f"Client id {client_id} is not in any group, please recheck the config"
        )

    # Select different LoRA configuration parameters
    if model_name in ["bert-base-cased"]:
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[target_module],
            bias="none",
        )
    elif model_name in [
        "Llama-2-7b-hf",
        "TinyLlama-1.1B-intermediate-step-1431k-3T",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
    ]:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[target_module],
            bias="none",
        )
    else:
        raise ValueError("Unsupported Models")

    print(f"Client {client_id} LoRA config: {lora_config}")
    return lora_config


def init_local_model(training_configs, train_dataset, loraConfig, model_name):
    """
    Initialize the local model with LoRA configuration.

    Args:
        training_configs: Configuration object containing model and training settings.
        train_dataset: Dataset used for training.
        loraConfig: LoRA configuration for fine-tuning.
        model_name (str): Name of the model.

    Returns:
        model: The initialized and LoRA-adapted model.
    """
    
    if model_name in ["bert-base-cased"]:
        num_labels = train_dataset.features["label"].num_classes
        class_names = train_dataset.features["label"].names
        # print(f"number of labels: {num_labels}")
        # print(f"the labels: {class_names}")
        # Create an id2label mapping, we will need this for our classifier
        id2label = {i: label for i, label in enumerate(class_names)}
        model = AutoModelForSequenceClassification.from_pretrained(
            training_configs.base_model_path, id2label=id2label
        )
        model = get_peft_model(model, loraConfig)
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
        # from peft import get_peft_model
        model = get_peft_model(model, loraConfig)
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model


def set_individual_model_state_dict(client_num, lora_config, local_model, edge_model):
    """
    Set the state dictionary (weights) for a specific client's local LoRA model
    using the shared edge model's state dictionary.

    Args:
        client_num (int): Client number.
        lora_config: LoRA configuration object.
        local_model: The local model for the client.
        edge_model: The edge/global model state dictionary.

    Raises:
        Exception: If no target modules are found in the edge model.
    """
    target_modules = lora_config.target_modules
    local_model_dict = get_peft_model_state_dict(local_model)
    layer_dict = {}

    for layer in local_model_dict.keys():
        if layer in edge_model:
            layer_dict[layer] = edge_model[layer]

    if not layer_dict:
        raise Exception("The local model has no available target module.")
    # print(f"client{client_num}`s layer dict is")
    # print(layer_dict.keys())
    set_peft_model_state_dict(local_model, layer_dict)
