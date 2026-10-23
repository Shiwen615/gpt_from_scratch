import argparse
import sys

import torch

import config
from data import CharDataset
from model import BigramLanguageModel
from train import train


def run_train(model, dataset):
    train(model, dataset)


def run_generate(model, dataset, to_stdout=False):
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    tokens = model.generate_stream(context, max_new_tokens=config.generate_tokens)
    if to_stdout:
        sys.stdout.reconfigure(encoding='utf-8')
        f = sys.stdout
    else:
        f = open(config.output_path, 'w', encoding='utf-8')
    count = 0
    try:
        # emit each character as soon as it is sampled instead of waiting for all of them
        for idx_next in tokens:
            piece = dataset.decode(idx_next[0].tolist())
            f.write(piece)
            f.flush()
            count += len(piece)
    finally:
        if not to_stdout:
            f.close()
    if to_stdout:
        print()
    else:
        print(f"wrote {count} characters to {config.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'generate'])
    parser.add_argument('--stdout', action='store_true',
                        help='generate mode: print the text to stdout instead of writing config.output_path')
    args = parser.parse_args()

    torch.manual_seed(config.seed)

    dataset = CharDataset()
    # keep stdout clean for generated text when --stdout is used
    info_stream = sys.stderr if args.stdout else sys.stdout
    print(f"text length: {len(dataset.text)}, vocab size: {dataset.vocab_size}", file=info_stream)

    model = BigramLanguageModel(dataset.vocab_size).to(config.device)
    if args.mode == 'train':
        run_train(model, dataset)
    else:
        run_generate(model, dataset, to_stdout=args.stdout)


if __name__ == '__main__':
    main()
