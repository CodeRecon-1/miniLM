import torch
import torch.nn as nn  # Neural network modules like Linear, Embedding, etc.
import torch.nn.functional as F  # Functional interface for operations like cross_entropy, silu, etc.
from torch.utils.data import Dataset, DataLoader  # Base class and utilities for loading datasets
from torch.cuda.amp import autocast, GradScaler  # 🔄 Automatic Mixed Precision (AMP) tools for faster/lower-memory training
from torch.optim import AdamW
import math  # Standard math operations (e.g. sqrt, exp, cos)
import random  # Python's random number utilities (used for seeding)
import numpy as np  # Numerical computing library, used for random seeding and general array ops

from datasets import load_dataset  # 🧁 Hugging Face Datasets library for streaming large datasets
from tqdm import tqdm  # ⏳ Progress bar visualization library, great for loops

import time  # ⌛ Timing utilities, measuring time
from transformers import AutoTokenizer  # 🤗 Load pretrained tokenizers from HuggingFace with one line

from dataclasses import dataclass  # 🧱 Define simple classes for configs with less boilerplate
from typing import List, Optional  # ✍️ Type hints for better readability and tooling

import warnings  # ⚠️ Suppress or handle warnings
import os  # 🗂️ File system operations (creating folders, path checking, etc.)
import pickle  # 💾 Python object serialization (used to save/load preprocessed datasets)

warnings.filterwarnings('ignore')  # Silences warnings for cleaner outputs during training
from utils.util_fn import repeat_kv
from config import ModelConfig


class Input_layer(torch.nn.Module):
  def __init__(self, config: ModelConfig):
    super(Input_layer, self).__init__()
    self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    # self.positional_embedding = torch.nn.Embedding(1024, config.d_model)
  def forward(self,x):
    return self.embedding(x) #+ self.positional_embedding(x)
  


class Attention(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.wq  = nn.Linear(config.d_model, config.d_model)
    self.wk = nn.Linear(config.d_model, config.d_model)
    self.wv= nn.Linear(config.d_model, config.d_model)
    self.dropout = nn.Dropout(0.1)
    self.norm = nn.LayerNorm(config.d_model)
    self.positional_embedding = nn.Embedding(config.max_seq_len, config.d_model)
    self.d_model = config.d_model # Store d_model for scaling
  def forward(self, x):
    # Apply positional embeddings
    positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
    x = x + self.positional_embedding(positions)

    q = self.wq(x)
    k = self.wk(x)
    v= self.wv(x)

    attn_scores = q.matmul(k.transpose(-2,-1))
    attn_weights = F.softmax(attn_scores/(self.d_model**0.5), dim=-1) # Use self.d_model
    attn_output = attn_weights.matmul(v)
    return attn_output
  
class FeedForward(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.norm = nn.LayerNorm(config.d_model) # Corrected LayerNorm initialization
    self.w1 = nn.Linear(config.d_model, 4*config.d_model)
    self.w2 = nn.Linear(4*config.d_model, config.d_model) # Output d_model for residual connection
    self.dropout = nn.Dropout(config.dropout)
  def forward(self, x):
    x= self.norm(x)
    x = self.w1(x)
    x = F.gelu(x) # Common activation function for FeedForward
    x = self.dropout(x)
    x = self.w2(x)
    return x


import torch.nn as nn
class TransformerBlock(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.attention = Attention(config)
    self.feed_forward = FeedForward(config)
    self.norm1 = nn.LayerNorm(config.d_model)
    self.norm2 = nn.LayerNorm(config.d_model)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    # Attention block with residual connection and layer norm
    attn_output = self.attention(self.norm1(x))
    x = x + self.dropout(attn_output) # Residual connection

    # Feed forward block with residual connection and layer norm
    ff_output = self.feed_forward(self.norm2(x))
    x = x + self.dropout(ff_output) # Residual connection
    return x


class SimpleTransformer(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.input_layer = Input_layer(config)
    self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
    self.output_head = nn.Linear(config.d_model, config.vocab_size)

  def forward(self, x):
    x = self.input_layer(x) # Embeddings
    for block in self.transformer_blocks:
      x = block(x)
    logits = self.output_head(x)
    return logits
