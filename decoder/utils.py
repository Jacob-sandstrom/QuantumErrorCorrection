#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and layers for the FEC package."""

import numpy as np
import yaml
import torch
import warnings
import sys


def parse_yaml(yaml_config):
    
    if yaml_config is not None:
        with open(yaml_config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    # default settings
    else:
        config = {}
        config["paths"] = {
            "root": "../",
            "save_dir": "../training_outputs",
            "model_name": "graph_decoder",
            "save_model_dir": "temp_dir",
            "saved_model_path": "temp_dir"
        }
        config["model_settings"] = {
            # "hidden_channels_GCN": [32, 128, 256, 512, 512, 256, 256],
            "hidden_channels_GCN": [32, 128, 256],
            "hidden_channels_MLP": [256, 128, 64],
            # "num_classes": 12
            "num_classes": 1
        }
        config["graph_settings"] = {
            "code_size": 3,
            "error_rate": 0.001,
            "m_nearest_nodes": 5,
            "d_t": 1,
            "train_error_rate": 0.2,
            "test_error_rate": 0.2
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["training_settings"] = {
            "seed": None,
            "dataset_size": 706381,
            "batch_size": 1000,
            "epochs": 1,
            "lr": 0.01,
            "device": device,
            "resume_training": False,
            "current_epoch": 0,
            "wandb": False,
            "validation_set_size": 1000,
            "test_set_size": 1000,
            "outcome_file": "dist3_time3_data/Outcome_data/outcome_dict_ibm_kyiv_simulator_3_706381_3_0.0.json",
            "syndromes_file": "dist3_time3_data/Detector_data/detector_dict_ibm_kyiv_simulator_3_706381_3_0.0.json"
        }
    
    # read settings into variables
    paths = config["paths"]
    model_settings = config["model_settings"]
    graph_settings = config["graph_settings"]
    training_settings = config["training_settings"]
    
    return paths, model_settings, graph_settings, training_settings
