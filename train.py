import torch

import config


@torch.no_grad()
def estimate_loss(model, dataset):
    """Average loss over eval_iters batches for each split."""
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            xb, yb = dataset.get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, dataset):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_val = float('inf')
    for step in range(config.max_iters):
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset)
            print(f"step {step}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
            if losses['val'] < best_val:
                best_val = losses['val']
                torch.save(model.state_dict(), config.checkpoint_path)
                print(f"  saved best weights (val loss = {best_val:.4f})")

        xb, yb = dataset.get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
