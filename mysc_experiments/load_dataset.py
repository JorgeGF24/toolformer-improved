from datasets import load_dataset
from torch.utils.data import DataLoader

cache_dir = "/vol/bitbucket/jg2619/data/low_perplexity/"


# Read data from csv files good_data/calculator/0.csv, good_data/calculator/1.csv, good_data/calculator/2.csv, ...
# Store csv files in Dataset object:
dataset = load_dataset("/vol/bitbucket/jg2619/data/low_perplexity/", cache_dir=cache_dir, split="train")
dataset.set_format("torch")
dataloader = DataLoader(dataset, batch_size=20)