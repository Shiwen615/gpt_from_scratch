import torch

seed = 1337
data_path = 'input.txt'
checkpoint_path = 'best.pt'
output_path = 'more.txt'
generate_tokens = 10000

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
num_heads = 6
num_layers = 6
dropout = 0.2
