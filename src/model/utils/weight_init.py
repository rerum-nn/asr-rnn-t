import math

import torch
import torch.nn as nn


def init_weights_xavier_uniform(module):
    """
    Initialize weights using Xavier uniform initialization.
    Good for most neural network layers.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.1)


def init_weights_xavier_normal(module):
    """
    Initialize weights using Xavier normal initialization.
    Alternative to uniform initialization.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.1)


def init_weights_he_uniform(module):
    """
    Initialize weights using He uniform initialization.
    Good for ReLU activations.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.1)


def init_weights_he_normal(module):
    """
    Initialize weights using He normal initialization.
    Alternative to uniform initialization for ReLU.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.1)


def init_weights_conformer_style(module):
    """
    Initialize weights specifically for Conformer architecture.
    Based on the original Conformer paper recommendations.
    """
    if isinstance(module, nn.Linear):
        # Use Xavier uniform for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        # Use He initialization for conv layers (good for ReLU)
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # Use He initialization for conv2d layers
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Use normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0, std=0.1)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm with ones and zeros
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        # Initialize LSTM weights
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def init_weights_rnnt_style(module):
    """
    Initialize weights specifically for RNN-T architecture.
    Based on RNN-T best practices.
    """
    if isinstance(module, nn.Linear):
        # Use smaller initialization for linear layers in RNN-T
        nn.init.normal_(module.weight, mean=0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        # Use He initialization for conv layers
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # Use He initialization for conv2d layers
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Use smaller initialization for embeddings
        nn.init.normal_(module.weight, mean=0, std=0.01)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm with ones and zeros
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        # Initialize LSTM weights with smaller std
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)


def init_weights_conservative(module):
    """
    Conservative initialization to prevent gradient explosion.
    Uses smaller standard deviations.
    """
    if isinstance(module, nn.Linear):
        # Very small initialization
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        # Small initialization for conv layers
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # Small initialization for conv2d layers
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Very small initialization for embeddings
        nn.init.normal_(module.weight, mean=0, std=0.005)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm with ones and zeros
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        # Initialize LSTM weights with very small std
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif "bias" in name:
                nn.init.zeros_(param)


def apply_weight_init(model, init_method="conformer"):
    """
    Apply weight initialization to the entire model.

    Args:
        model: PyTorch model
        init_method: str, one of ['conformer', 'rnnt', 'conservative', 'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal']
    """
    init_functions = {
        "conformer": init_weights_conformer_style,
        "rnnt": init_weights_rnnt_style,
        "conservative": init_weights_conservative,
        "xavier_uniform": init_weights_xavier_uniform,
        "xavier_normal": init_weights_xavier_normal,
        "he_uniform": init_weights_he_uniform,
        "he_normal": init_weights_he_normal,
    }

    if init_method not in init_functions:
        raise ValueError(
            f"Unknown init_method: {init_method}. Available: {list(init_functions.keys())}"
        )

    init_func = init_functions[init_method]

    # Apply initialization to all modules
    for module in model.modules():
        init_func(module)

    print(f"Weight initialization applied using method: {init_method}")
