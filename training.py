import torch
import torch.nn as nn
from config import ModelConfig
from model import SimpleTransformer
from loader import TextTokenDataset , load_and_cache_data
# --- Training Configuration ---
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
from torch.optim import AdamW
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=128, # Reduced for a "small" model and faster training
    n_heads=4,
    n_layers=2,
    d_ff=512,
    dropout=0.1,
    max_seq_len=64 # Reduced sequence length for faster training
)

# Instantiate the model
transformer_model = SimpleTransformer(config)

# Prepare dataset and dataloader


# Optimizer and Loss Function
optimizer = AdamW(transformer_model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()#ignore_index=tokenizer_id).pad_token

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model.to(device)

print(f"Model configured with vocab_size={config.vocab_size}, d_model={config.d_model}, n_layers={config.n_layers}")
print(f"Using device: {device}")


from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import tqdm

texts, tokenizer, tokens = load_and_cache_data()
dataset = TextTokenDataset(tokens, seq_len=5)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


def train_model(model, dataloader, optimizer, loss_fn, device, epochs=5):
  model.train() # Set the model to training mode
  for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      optimizer.zero_grad() # Clear previous gradients

      with autocast(enabled=torch.cuda.is_available()): # Enable AMP for mixed precision training
        logits = model(x_batch)
        # Reshape logits and targets for CrossEntropyLoss
        # logits: (batch_size, seq_len, vocab_size)
        # y_batch: (batch_size, seq_len)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))

      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      if batch_idx % 100 == 0: # Print every 100 batches
        tqdm.write(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# Run training
train_model(transformer_model, dataloader, optimizer, loss_fn, device, epochs=1)
#SAVE the model
torch.save(transformer_model.state_dict(), 'model_weights.pth')
