import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from server import (
    init_global_model,
    edge_aggregate_layer,
    split_layers,
    global_aggregate_3matrix,
    global_aggregate_4matrix,
)
from config import save_config, get_args_config
from dataset import preprocess_dateset, split_dataset
from trainer import get_trainer
from client import (
    init_local_model,
    set_individual_model_state_dict,
    generate_lora_config,
)
from utils import (
    get_layer_importance_ranking,
    freeze_layers,
    pruning_groups,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
)

llama_family = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-2-7b-hf",
    "TinyLlama-1.1B-intermediate-step-1431k-3T",
]
bert_family = ["bert-base-cased"]

# ===== get training config =====
training_configs = get_args_config()
save_config(training_configs)
model_name = training_configs.model_name
freeze_layers_num = training_configs.freeze_layers_num

k_group_clients = training_configs.k_group_clients
q_group_clients = training_configs.q_group_clients
v_group_clients = training_configs.v_group_clients
# for llama and tiny llama
if model_name in llama_family:
    o_group_clients = training_configs.o_group_clients

edge_clients = training_configs.edge_clients


# ===== Load and splite the dataset =====
train_dataset, eval_dataset = preprocess_dateset()
local_train_datasets, local_eval_datasets = split_dataset(train_dataset, eval_dataset)
sample_num_list = [
    len(local_train_datasets[i])
    for i in range(training_configs.num_clients + 1)  # the last client is global server
]  # for fedeavg aggregation
print(f"sample_num_list:{sample_num_list}")

# ===== init FL global model and distribute to clients =====
model, global_dict = init_global_model(training_configs, train_dataset, model_name)
save_dir = "clients_peft_states"
os.makedirs(save_dir, exist_ok=True)  # make sure the dir is exit
for client_id in range(training_configs.num_clients):
    file_path = os.path.join(save_dir, f"lora_checkpoint_client_{client_id}.pt")
    torch.save(global_dict, file_path)  # save state_dict
    print(f"Begining round: client {client_id} lora model saved to {file_path}")


# the form of sorted_layers is like:
# [[0, 1, 2],
#  [0, 1, 2],
#  [0, 1, 2],
#  ...
# ]
sorted_layers = [
    list(range(NUM_HIDDEN_LAYERS)) for _ in range(training_configs.num_clients)
]  # init sorted_layers of each client

# the form of layer_importance is like:
# [
# {0: 0.123, 1: 0.456, 2: 0.789},
# {0: 0.321, 1: 0.654, 2: 0.987},
# ...
# ]
layer_importance = [
    {layer_id: 0.0 for layer_id in range(NUM_HIDDEN_LAYERS)}
    for _ in range(training_configs.num_clients)
]


# ===== Start federated training =====
k_group_training_loss = [[] for i in range(len(k_group_clients))]
q_group_training_loss = [[] for i in range(len(q_group_clients))]
v_group_training_loss = [[] for i in range(len(v_group_clients))]
# for llama and tiny llama
if model_name in llama_family:
    o_group_training_loss = [[] for i in range(len(o_group_clients))]

for global_round in range(training_configs.num_rounds):  # main loop of LLM FL training

    # ====== init separate dicts k, q, v (and o) =====
    if model_name in bert_family:
        clients_this_round = (
            k_group_clients + q_group_clients + v_group_clients
        )  # client id start from 0
        k_lora, q_lora, v_lora = split_layers(
            global_dict, model_name
        )  # Before setting state dict, we need to process global dict into separate dicts of k, q and v
    elif model_name in llama_family:
        clients_this_round = (
            k_group_clients + q_group_clients + v_group_clients + o_group_clients
        )  # client id start from 0
        k_lora, q_lora, v_lora, o_lora = split_layers(
            global_dict, model_name
        )  # Before setting state dict, we need to process global dict into separate dicts of k, q, v and o
    print(
        f">> ==================== Global round {global_round+1} : {clients_this_round} ===================="
    )

    # ===== edge server aggregating round =====
    for edge_round in range(training_configs.edge_aggregate_round):
        print(f">> ==================== Edge round {edge_round+1} ====================")
        # ===== Start o group training =====
        if model_name in llama_family:
            print(">> ==================== start o group training ====================")
            for client in o_group_clients:
                loraconfig = generate_lora_config(client)
                local_model = init_local_model(
                    training_configs, train_dataset, loraconfig, model_name
                )  # every client has its own local model
                set_individual_model_state_dict(client, loraconfig, local_model, o_lora)

                # ===== Layer Freezing and Training =====
                print(
                    f"client #{client} sorted layers: {sorted_layers[client]}\n layer importance: {layer_importance[client]}"
                )
                if client in edge_clients:
                    # The edge server does not need to freeze layers
                    layers_name_to_freeze, layers_id_to_freeze = (
                        [],
                        [],
                    )  # The edge server does not need to freeze layers
                else:
                    layers_name_to_freeze, layers_id_to_freeze = freeze_layers(
                        sorted_layers[client], local_model, freeze_layers_num
                    )
                trainer = get_trainer(
                    training_configs,
                    local_model,
                    local_train_datasets[client],
                    local_eval_datasets[client],
                    model_name,
                    layers_name_to_freeze,
                )
                results, sensitive_dict = trainer.train()

                # ===== Next round of layer sorting =====
                sorted_layers[client], layer_importance[client] = (
                    get_layer_importance_ranking(sensitive_dict, layers_id_to_freeze)
                )  # The ranking of the next round is determined by the training results of this round

                o_group_training_loss[
                    client
                    - len(k_group_clients)
                    - len(q_group_clients)
                    - len(v_group_clients)
                ].append(results.training_loss)

                # ===== Save the clients` local state dict =====
                local_model_file_path = os.path.join(
                    save_dir, f"lora_checkpoint_client_{client}.pt"
                )
                torch.save(
                    get_peft_model_state_dict(local_model), local_model_file_path
                )  # save lora model

            o_edge_dict = edge_aggregate_layer(save_dir, o_group_clients)
            set_peft_model_state_dict(model, o_edge_dict)  # Update o edge model

            # ===== Save the last edge model =====
            # if edge_round == training_configs.edge_aggregate_round - 1:
            #     trainer.save_model(
            #         os.path.join(
            #             training_configs.output_dir,
            #             f"checkpoint-global-{global_round+1}-edge-o",
            #         )
            #     )
            #     np.save(
            #         os.path.join(
            #             training_configs.output_dir, "o_group_training_loss.npy"
            #         ),
            #         np.array(o_group_training_loss),
            #     )

        # ===== Start k group training =====
        print(">> ==================== start k group training ====================")
        for client in k_group_clients:
            loraconfig = generate_lora_config(client)

            local_model = init_local_model(
                training_configs, train_dataset, loraconfig, model_name
            )  # every client has its own local model
            set_individual_model_state_dict(client, loraconfig, local_model, k_lora)

            # ===== Layer Freezing and Training =====
            print(
                f"client #{client} sorted layers: {sorted_layers[client]}\n layer importance: {layer_importance[client]}"
            )
            if client in edge_clients:
                # The edge server does not need to freeze layers
                layers_name_to_freeze, layers_id_to_freeze = (
                    [],
                    [],
                )  # The edge server does not need to freeze layers
            else:
                layers_name_to_freeze, layers_id_to_freeze = freeze_layers(
                    sorted_layers[client], local_model, freeze_layers_num
                )
            trainer = get_trainer(
                training_configs,
                local_model,
                local_train_datasets[client],
                local_eval_datasets[client],
                model_name,
                layers_name_to_freeze,
            )
            results, sensitive_dict = trainer.train()

            # ===== Next round of layer sorting =====
            sorted_layers[client], layer_importance[client] = (
                get_layer_importance_ranking(sensitive_dict, layers_id_to_freeze)
            )  # The ranking of the next round is determined by the training results of this round

            k_group_training_loss[client].append(results.training_loss)

            # ===== Save the clients` local model =====
            local_model_file_path = os.path.join(
                save_dir, f"lora_checkpoint_client_{client}.pt"
            )
            torch.save(
                get_peft_model_state_dict(local_model), local_model_file_path
            )  # save state_dict

        k_edge_dict = edge_aggregate_layer(save_dir, k_group_clients)
        set_peft_model_state_dict(model, k_edge_dict)  # Update k edge model
        # Save the last edge model
        # if edge_round == training_configs.edge_aggregate_round - 1:
        #     trainer.save_model(
        #         os.path.join(
        #             training_configs.output_dir,
        #             f"checkpoint-global-{global_round+1}-edge-k",
        #         )
        #     )
        #     np.save(
        #         os.path.join(training_configs.output_dir, "k_group_training_loss.npy"),
        #         np.array(k_group_training_loss),
        #     )

        # ===== Start q group training =====
        print(">> ==================== start q group training ====================")
        for client in q_group_clients:
            loraconfig = generate_lora_config(client)
            local_model = init_local_model(
                training_configs, train_dataset, loraconfig, model_name
            )  # every client has its own local model
            set_individual_model_state_dict(client, loraconfig, local_model, q_lora)

            # ===== Layer Freezing and Training =====
            print(
                f"client #{client} sorted layers: {sorted_layers[client]}\n layer importance: {layer_importance[client]}"
            )
            if client in edge_clients:
                # The edge server does not need to freeze layers
                layers_name_to_freeze, layers_id_to_freeze = (
                    [],
                    [],
                )  # The edge server does not need to freeze layers
            else:
                layers_name_to_freeze, layers_id_to_freeze = freeze_layers(
                    sorted_layers[client], local_model, freeze_layers_num
                )
            trainer = get_trainer(
                training_configs,
                local_model,
                local_train_datasets[client],
                local_eval_datasets[client],
                model_name,
                layers_name_to_freeze,
            )
            results, sensitive_dict = trainer.train()

            # ===== Next round of layer sorting =====
            sorted_layers[client], layer_importance[client] = (
                get_layer_importance_ranking(sensitive_dict, layers_id_to_freeze)
            )  # The ranking of the next round is determined by the training results of this round

            q_group_training_loss[client - len(k_group_clients)].append(
                results.training_loss
            )

            # ===== Save the clients` local model =====
            local_model_file_path = os.path.join(
                save_dir, f"lora_checkpoint_client_{client}.pt"
            )
            torch.save(
                get_peft_model_state_dict(local_model), local_model_file_path
            )  # save state_dict

        q_edge_dict = edge_aggregate_layer(save_dir, q_group_clients)
        set_peft_model_state_dict(model, q_edge_dict)  # Update q edge model

        # Save the last edge model
        # if edge_round == training_configs.edge_aggregate_round - 1:
        #     trainer.save_model(
        #         os.path.join(
        #             training_configs.output_dir,
        #             f"checkpoint-global-{global_round+1}-edge-q",
        #         )
        #     )
        #     np.save(
        #         os.path.join(training_configs.output_dir, "q_group_training_loss.npy"),
        #         np.array(q_group_training_loss),
        #     )

        # ===== Start v group training =====
        print(">> ==================== start v group training ====================")
        for client in v_group_clients:
            loraconfig = generate_lora_config(client)
            local_model = init_local_model(
                training_configs, train_dataset, loraconfig, model_name
            )  # every client has its own local model
            set_individual_model_state_dict(client, loraconfig, local_model, v_lora)

            # ===== Layer Freezing and Training =====
            print(
                f"client #{client} sorted layers: {sorted_layers[client]}\n layer importance: {layer_importance[client]}"
            )
            if client in edge_clients:
                # The edge server does not need to freeze layers
                layers_name_to_freeze, layers_id_to_freeze = (
                    [],
                    [],
                )  # The edge server does not need to freeze layers
            else:
                layers_name_to_freeze, layers_id_to_freeze = freeze_layers(
                    sorted_layers[client], local_model, freeze_layers_num
                )
            trainer = get_trainer(
                training_configs,
                local_model,
                local_train_datasets[client],
                local_eval_datasets[client],
                model_name,
                layers_name_to_freeze,
            )
            results, sensitive_dict = trainer.train()

            # ===== Next round of layer sorting =====
            sorted_layers[client], layer_importance[client] = (
                get_layer_importance_ranking(sensitive_dict, layers_id_to_freeze)
            )  # The ranking of the next round is determined by the training results of this round

            v_group_training_loss[
                client - len(k_group_clients) - len(q_group_clients)
            ].append(results.training_loss)

            # ===== Save the clients` local model =====
            local_model_file_path = os.path.join(
                save_dir, f"lora_checkpoint_client_{client}.pt"
            )
            torch.save(
                get_peft_model_state_dict(local_model), local_model_file_path
            )  # save state_dict

        v_edge_dict = edge_aggregate_layer(save_dir, v_group_clients)
        set_peft_model_state_dict(model, v_edge_dict)  # Update q edge model

        # Save the last edge model
        # if edge_round == training_configs.edge_aggregate_round - 1:
        #     trainer.save_model(
        #         os.path.join(
        #             training_configs.output_dir,
        #             f"checkpoint-global-{global_round+1}-edge-v",
        #         )
        #     )
        #     np.save(
        #         os.path.join(training_configs.output_dir, "v_group_training_loss.npy"),
        #         np.array(v_group_training_loss),
        #     )

    # ===== Server aggregates the local models =====
    if model_name in bert_family:
        global_dict = global_aggregate_3matrix(k_edge_dict, q_edge_dict, v_edge_dict)
    elif model_name in llama_family:
        global_dict = global_aggregate_4matrix(
            k_edge_dict, q_edge_dict, v_edge_dict, o_edge_dict
        )

    set_peft_model_state_dict(model, global_dict)  # Update global model

    # ===== Starting global training =====
    print(">> ==================== start global training ====================")
    layers_name_to_freeze = []
    trainer = get_trainer(
        training_configs,
        model,
        local_train_datasets[training_configs.num_clients],
        local_eval_datasets[training_configs.num_clients],
        model_name,
        layers_name_to_freeze,
    )
    results, sensitive_dict = trainer.train()
    global_dict = get_peft_model_state_dict(trainer.model)

    # ===== Save the global model =====
    trainer.save_model(
        os.path.join(training_configs.output_dir, f"checkpoint-{global_round+1}")
    )
