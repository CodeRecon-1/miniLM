from dataclasses import dataclass
# from torch.utils.data import Dataset, DataLoader
@dataclass
class ModelConfig:
  vocab_size: int = 49152 # This will be updated from tokenizer.vocab_size
  d_model: int = 512
  n_heads: int = 8
  n_layers: int = 1
  d_ff: int = 2048
  dropout: float = 0.1
  max_seq_len: int = 512 # Maximum sequence length for positional embeddings
  attention_bias: bool = True
  rms_norm_eps: float = 1e-6