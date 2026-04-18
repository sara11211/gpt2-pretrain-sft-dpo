import torch
import torch.nn as nn


# =========================================================
# Approach 1: Simple Implementation
# =========================================================

class CausalSelfAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        
        # Initialize weight matrices 
        self.W_query  = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Register a buffer non-trainable tensor 
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  
        
        
    def forward(self, x ):
        b, num_tokens, d_in = x.shape   # x = batch_size, num_tokens, embedding_dim
        
        # Project inputs to query, key, value spaces
        queries = self.W_query(x)
        keys    = self.W_key(x)    
        values  = self.W_value(x)
        
        # Compute attention scores and weights 
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors for all tokens
        context_vec = attn_weights @ values
        return context_vec 


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False, num_heads=2):
        super().__init__()
        
        # Create a list of single attention heads
        self.heads = nn.ModuleList([
            CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        
        # (batch_size, sequence_length, head_dim)
        # Run each head independently and concatenate along last dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)


# =========================================================
# Approach 2: Alternative Implementation
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # dimension of each head's output

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # (b, num_tokens, d_in) -> (b, num_tokens, d_out)
        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the matrices into multiple heads
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute self-attentino with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec



# =========================================================
# Main Execution 
# =========================================================

if __name__ == "__main__":

    torch.manual_seed(123)

    context_length = 4
    output_dim = 256

    input_embeddings = torch.randn(8, 4, output_dim)

    # -------- Approach 1 --------
    print("Running Approach 1")

    num_heads = 2
    d_in = output_dim
    d_out = d_in // num_heads

    mha_wrapper = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads
    )

    context_vecs = mha_wrapper(input_embeddings)
    print("Approach 1 output shape:", context_vecs.shape)

    # -------- Approach 2 --------
    print("\nRunning Approach 2")

    mha = MultiHeadAttention(
        d_in, output_dim, context_length, 0.0, num_heads=2
    )

    context_vecs = mha(input_embeddings)
    print("Approach 2 output shape:", context_vecs.shape)