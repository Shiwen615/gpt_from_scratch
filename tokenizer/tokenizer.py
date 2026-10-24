import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='path to the input text file')
input_path = parser.parse_args().input_path

vocab_size = 512 # the desired final vocabulary size
num_merges = vocab_size - 256

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

with open(input_path, 'r', encoding='utf-8') as file:
    text = file.read()
text = text
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair}({stats[pair]} times) into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

# JSON keys must be strings, so store each (a, b) pair as "a,b"
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encoder.json')
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump({f"{a},{b}": idx for (a, b), idx in merges.items()}, file, indent=2)
print("saved merges to", output_path)

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")