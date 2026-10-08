import torch

import config


class CharDataset:
    """Character-level tokenizer plus train/val split and batch sampling."""

    def __init__(self, path=config.data_path, train_frac=0.9):
        with open(path, 'r', encoding='utf-8') as file:
            self.text = file.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(config.device), y.to(config.device)
