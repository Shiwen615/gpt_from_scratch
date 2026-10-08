import argparse

import torch

import config
from data import CharDataset
from model import BigramLanguageModel
from train import train


def run_train(model, dataset):
    train(model, dataset)


def run_generate(model, dataset):
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    out = model.generate(context, max_new_tokens=config.generate_tokens)[0].tolist()
    text = dataset.decode(out)
    with open(config.output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"wrote {len(text)} characters to {config.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'generate'])
    args = parser.parse_args()

    torch.manual_seed(config.seed)

    dataset = CharDataset()
    print(f"text length: {len(dataset.text)}, vocab size: {dataset.vocab_size}")

    model = BigramLanguageModel(dataset.vocab_size).to(config.device)
    if args.mode == 'train':
        run_train(model, dataset)
    else:
        run_generate(model, dataset)


if __name__ == '__main__':
    main()
