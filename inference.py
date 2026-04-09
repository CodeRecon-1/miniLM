import torch
from model import SimpleTransformer
from config import ModelConfig
import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ModelConfig
def load_model_weights():
  transformer_model = SimpleTransformer(config)
  transformer_model.load_state_dict(torch.load('model_weights.pth'))

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, seq_len=config.max_seq_len, device=device):
  model.eval() # Set the model to evaluation mode
  encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
  input_ids = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)

  generated_tokens = encoded_prompt

  with torch.no_grad(): # Disable gradient calculations
    for _ in tqdm(range(max_new_tokens), desc="Generating text"):
      # Limit input to the model's sequence length if it exceeds
      current_input_ids = input_ids[:, -seq_len:]

      # Forward pass to get logits
      with autocast(enabled=torch.cuda.is_available()):
        logits = model(current_input_ids)

      # Get logits for the last token and apply temperature
      logits = logits[:, -1, :] / temperature
      # Apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1)
      # Sample the next token
      next_token_id = torch.multinomial(probs, num_samples=1)

      # Append to input for the next iteration
      input_ids = torch.cat([input_ids, next_token_id], dim=1)
      generated_tokens.append(next_token_id.item())

  return tokenizer.decode(generated_tokens)

# Example Usage:
# prompt = "Once upon a time, there was a small cat who loved to "
# generated_story = generate_text(transformer_model, tokenizer, prompt, max_new_tokens=100)
# print("Generated Text:")
# print(generated_story)
