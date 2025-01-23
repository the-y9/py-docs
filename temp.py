import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

#%% Tokenization and Encoding
def tokenize_and_encode(text):
    tokens = text.split()
    token_ids = {token: idx for idx, token in enumerate(set(tokens))}
    encoded = [token_ids[token] for token in tokens]
    return tokens, token_ids, encoded

#%% Create Embedding Matrix
def create_embedding_matrix(encoded, embedding_dim=64):
    return torch.randn(size=(len(encoded), embedding_dim))

#%% Positional Encoding
def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = torch.sin(torch.tensor(pos) / (10000 ** (i / embed_dim)))  # Convert pos to tensor
            if i + 1 < embed_dim:
                pe[pos, i + 1] = torch.cos(torch.tensor(pos) / (10000 ** (i / embed_dim)))  # Convert pos to tensor
    return pe


#%%

class MHA(nn.Module):
    def __init__(self, dmodel, heads):
        super(MHA, self).__init__()
        self.dk = dmodel // heads
        self.heads = heads
        self.Wq = nn.Parameter(torch.randn(heads, dmodel, self.dk))
        self.Wk = nn.Parameter(torch.randn(heads, dmodel, self.dk))
        self.Wv = nn.Parameter(torch.randn(heads, dmodel, self.dk))
        self.Wo = nn.Parameter(torch.randn(dmodel, dmodel))

    def forward(self, H):
        # H is expected to be of shape (batch_size, seq_len, dmodel)
        batch_size, seq_len, _ = H.size()

        Q = torch.einsum('btd,hdq->bhq', H, self.Wq)  # Shape: (batch_size, heads, seq_len, dk)
        K = torch.einsum('btd,hdq->bhk', H, self.Wk)  # Shape: (batch_size, heads, seq_len, dk)
        V = torch.einsum('btd,hdv->bhv', H, self.Wv)  # Shape: (batch_size, heads, seq_len, dk)

        # Update: Using the correct dimensions in einsum
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / torch.sqrt(torch.tensor(self.dk, dtype=H.dtype))  # Ensure dk is a tensor
        soft_attn = torch.softmax(attn_scores, dim=-1)  # Shape: (batch_size, heads, seq_len, seq_len)

        attn = torch.einsum('bhqk,bhvd->bhqv', soft_attn, V)  # Shape: (batch_size, heads, seq_len, dk)

        # Reshape and project back to dmodel
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, dmodel)
        output = torch.einsum('btd,tdm->btd', attn, self.Wo)  # Shape: (batch_size, seq_len, dmodel)

        return output


#%% Layer Normalization
def layer_normalization(v, gamma=0.1, beta=0.01):
    mu = torch.mean(v, dim=-1, keepdim=True)
    var = torch.var(v, dim=-1, keepdim=True)
    normalized = (v - mu) / torch.sqrt(var + 1e-6)
    return gamma * normalized + beta

#%% Feedforward Network
def feedforward_network(x):
    in_dim = x.size(2)
    x = nn.Linear(in_dim, 128)(x)
    x = nn.ReLU()(x)
    return nn.Linear(128, in_dim)(x)

#%% Sequence Generation
def generate_sequence(encoder_output, projection_layer, max_length=10):
    last_hidden_state = encoder_output[:, -1, :].unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
    generated_sequence = []

    for _ in range(max_length):
        logits = projection_layer(last_hidden_state)  # Shape: [batch_size, 1, vocab_size]
        probabilities = torch.softmax(logits, dim=-1)  # Shape: [batch_size, 1, vocab_size]
        predicted_idx = torch.argmax(probabilities, dim=-1)  # Shape: [batch_size, 1]
        generated_sequence.append(predicted_idx.item())
        last_hidden_state = encoder_output[:, predicted_idx.item(), :].unsqueeze(1)  # Use predicted index for next input

    return generated_sequence

#%% Main Execution
if __name__ == "__main__":
    # Input text for tokenization and encoding
    text = '''
    cat sat on couch
    mat is red
    cat is white
    couch is brown
    dog sat on mat
    dog is golden
    '''

    # Tokenization and encoding
    tokens, token_ids, encoded = tokenize_and_encode(text)

    # Create the embedding matrix
    embedding_matrix = create_embedding_matrix(encoded)

    # Positional Encoding
    positional_matrix = positional_encoding(len(encoded), embedding_matrix.size(1))

    # Input matrix for attention
    input_matrix = embedding_matrix + positional_matrix

    # Visualize heatmaps
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.imshow(embedding_matrix.detach().numpy())
    plt.colorbar(label="Encoding Value")
    plt.title("Embedding Matrix Heatmap")

    plt.subplot(3, 1, 2)
    plt.imshow(positional_matrix.detach().numpy())
    plt.colorbar(label="Encoding Value")
    plt.title("Positional Encoding Heatmap")

    plt.subplot(3, 1, 3)
    plt.imshow(input_matrix.detach().numpy())
    plt.colorbar(label="Encoding Value")
    plt.title("Input Matrix (Embedding + Positional) Heatmap")

    plt.tight_layout()
    plt.show()

    # Initialize Multi-Head Attention
    d_model = 64
    num_heads = 8
    # Initialize MHA
    mha = MHA(dmodel=64, heads=8)

    # Assume input_matrix is already defined and has the appropriate shape
    mha_out = mha(torch.tensor(input_matrix).unsqueeze(0))  # Add batch dimension

    # Normalize and apply feedforward network
    norm_mha = layer_normalization(input_matrix + mha_out.squeeze(0))
    ffn_out = feedforward_network(norm_mha.unsqueeze(0))

    # Final normalization
    norm_ffn = layer_normalization(ffn_out.squeeze(0) + norm_mha)

    # Projection layer for sequence generation
    vocab_size = len(token_ids)
    projection_layer = nn.Linear(d_model, vocab_size)

    # Generate a sequence
    generated_sequence = generate_sequence(norm_ffn.unsqueeze(0), projection_layer, max_length=10)

    # Convert indices back to words
    idx_to_word = {i: token for i, token in enumerate(token_ids.keys())}  # Mapping indices back to words
    generated_words = [idx_to_word[idx] for idx in generated_sequence]

    print("Generated sequence:", " ".join(generated_words))
