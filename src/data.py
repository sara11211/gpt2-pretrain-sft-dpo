import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


file_path = "../data/input.txt"

# Hyperparameters
batch_size = 8
max_length = 4
vocab_size = 50257
output_dim = 256
context_length = 1024


# =========================================================
# Dataset
# =========================================================

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Get the number of training exemples
    def __len__(self):
        return len(self.input_ids)

    # Get one training exemple
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# =========================================================
# DataLoader
# =========================================================

def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader



# =========================================================
# Main Execution
# =========================================================

def main():
    
    # Open dataset file and read the text
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Create token and position embedding layers
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # Create dataloader
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )

    # Iterate through the dataloader and get a batch of input and target 
    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        break

    # Print the shape of the input embeddings
    print("Input embeddings shape:", input_embeddings.shape)


if __name__ == "__main__":
    main()