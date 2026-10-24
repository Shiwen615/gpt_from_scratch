import json
import os

import torch

import config


class _Dataset:
    """Train/val split and batch sampling; subclasses provide encode/decode."""

    def _split(self, ids, train_frac):
        data = torch.tensor(ids, dtype=torch.long)
        n = int(train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(config.device), y.to(config.device)


class CharDataset(_Dataset):
    """Character-level tokenizer."""

    def __init__(self, path=config.data_path, train_frac=0.9):
        with open(path, 'r', encoding='utf-8') as file:
            self.text = file.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self._split(self.encode(self.text), train_frac)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


class BPEDataset(_Dataset):
    """Byte-level BPE tokenizer (see tokenizer/tokenizer.py) using merges from encoder.json."""

    def __init__(self, path=config.data_path, encoder_path=config.encoder_path, train_frac=0.9):
        with open(path, 'r', encoding='utf-8') as file:
            self.text = file.read()

        # JSON keys are "a,b" strings; dicts keep insertion order, which is the merge order
        with open(encoder_path, 'r', encoding='utf-8') as file:
            self.merges = {tuple(map(int, k.split(','))): idx for k, idx in json.load(file).items()}

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.vocab_size = len(self.vocab)

        self._split(self._load_ids(path, encoder_path), train_frac)

    def _load_ids(self, path, encoder_path):
        # encoding the full corpus in pure Python is slow, so cache it next to the encoder
        cache = encoder_path + '.ids.pt'
        sources = (path, encoder_path)
        if os.path.exists(cache) and os.path.getmtime(cache) >= max(map(os.path.getmtime, sources)):
            return torch.load(cache)
        ids = self.encode(self.text)
        torch.save(ids, cache)
        return ids

    @staticmethod
    def _merge(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, s):
        ids = list(s.encode('utf-8'))
        # apply merges in the order they were learned
        for pair, idx in self.merges.items():
            if len(ids) < 2:
                break
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')
