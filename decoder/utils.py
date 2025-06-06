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
            "save_model_dir": "training_outputs",
            "saved_model_path": "training_outputs"
        }
        config["model_settings"] = {
            # "hidden_channels_GCN": [32, 128, 256, 512, 512, 256, 256],
            # "hidden_channels_GCN": [32, 128, 256, 512, 256],
            "hidden_channels_GCN": [32, 128, 256],
            "hidden_channels_MLP": [256, 128, 64],
            # "num_classes": 12
            "num_classes": 1
        }
        config["graph_settings"] = {
            "error_rate": 0.001,
            "m_nearest_nodes": 5,
            "train_error_rate": 0.2,
            "test_error_rate": 0.2
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["training_settings"] = {
            "seed": None,
            "batch_size": 5000,
            "lr": 0.01,
            "device": device,
            "resume_training": False,
            "current_epoch": 0,
            "test_set_size": 1000,
            "training_folder": "data/d3_t3_torino"
        }
    
    # read settings into variables
    paths = config["paths"]
    model_settings = config["model_settings"]
    graph_settings = config["graph_settings"]
    training_settings = config["training_settings"]
    
    return paths, model_settings, graph_settings, training_settings
