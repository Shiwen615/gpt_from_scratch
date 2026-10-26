import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            'tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cache=None):
        # cache: a dict holding this head's past keys/values (None = no caching, used in training).
        # x only contains the NEW tokens; the old ones are already in the cache.
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        if cache is not None:
            if 'k' in cache:  # append the new k, v after the cached ones
                k = torch.cat([cache['k'], k], dim=1)
                v = torch.cat([cache['v'], v], dim=1)
            cache['k'], cache['v'] = k, v
        S = k.shape[1]  # total length = cached tokens + new tokens
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # the new queries sit at positions S-T ... S-1, so use those rows of the mask
        wei = wei.masked_fill(self.tril[S - T:S, :S] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cache=None):
        # cache: one dict per head
        x = torch.cat([h(x, cache[i] if cache else None) for i, h in enumerate(self.heads)], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa_head = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, cache=None):
        x = x + self.sa_head(self.ln1(x), cache)
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        n_embd = config.n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, config.num_heads) for _ in range(config.num_layers)],
                                    nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def new_cache(self):
        """ an empty KV cache: cache[layer][head] is a dict that Head fills with its keys/values """
        return [[{} for _ in range(config.num_heads)] for _ in range(config.num_layers)]

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        # with a cache, idx holds only the new tokens, so their positions start after the cached ones
        start = cache[0][0]['k'].shape[1] if cache and cache[0][0] else 0
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(start, start + T, device=idx.device))
        x = tok_emb + pos_emb
        # self.blocks = [Block x num_layers, final LayerNorm] (kept as one Sequential so old checkpoints load)
        for i, block in enumerate(self.blocks[:-1]):
            x = block(x, cache[i] if cache else None)
        x = self.blocks[-1](x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens):
        """ yield each new token (shape (B, 1)) as soon as it is sampled, using a KV cache """
        cache = self.new_cache()
        idx_cond = idx[:, -config.block_size:]  # first step: feed the whole prompt, later steps: 1 token
        for _ in range(max_new_tokens):
            logits, _ = self(idx_cond, cache=cache)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
            if cache[0][0]['k'].shape[1] + 1 > config.block_size:
                # position table only has block_size entries and the cache is full:
                # drop it and rebuild from the most recent half of the window
                cache = self.new_cache()
                idx_cond = idx[:, -config.block_size // 2:]
            else:
                idx_cond = idx_next  # only the newest token needs computing; the rest is cached

    def generate(self, idx, max_new_tokens):
        for idx_next in self.generate_stream(idx, max_new_tokens):
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

