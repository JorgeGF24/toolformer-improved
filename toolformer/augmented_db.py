from csv import QUOTE_ALL, DictWriter, QUOTE_MINIMAL
import os
import logging
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPTJConfig, LlamaConfig
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from beartype import beartype



from toolformer import Toolformer

FILE_BATCH_SIZE = 1   # number of files to load at a time
GENERATION_TARGET = 50000                                                       
MAX_FILE_SIZE = 2000

file_batch = 0
stats_num = 0
data_loader = None

@beartype
def augment_db(
    dataset_id: str,
    tool_name: str,
    augment_dir,
    data_batch_size=1000,
    num_of_processed_batches=0,
    custom_dataset=None,
    skip_files:list=[],
    model_name:str="GPTJ",  # "GPTJ" or "LLAMA",
    experiment_config:dict={},
    generated_so_far=0,
    device=torch.device('cuda'),
    world_size=1,
    **kwargs,
):
    global data_loader, file_batch, stats_num
    dataset_id = dataset_id.lower()
    model_name = model_name.upper()

    data_batch_size = kwargs.get("prompt_batch_size", data_batch_size//10)*10

    file_batch = 0
    os.makedirs(augment_dir + "/stats", exist_ok=True)
    stats_num = len([file for file in os.listdir(os.path.join(augment_dir,"stats")) if file.startswith("stats")])

            
    # Task should have tf_kwargs with:
    # max_args_length, max_data_length, response_length, m_arg_samples, prompt_batch_size, filtering_batch_size, raw_tool_prompt, tool_name, tool, tool_check_duplicates, debug_level
    # Initialize the Toolformer
    toolformer = Toolformer(
        model_name=model_name,
        tool_name=tool_name,
        log_dir=augment_dir + "/logs",
        experiment_config=experiment_config,
        using_llama= model_name == "LLAMA",
        device=device,
        world_size=world_size,
        **kwargs
    )
    
    load_data_kwargs = {
        "dataset_id": dataset_id,
        "tool_name": tool_name,
        "data_batch_size": data_batch_size,
        "skip_files": skip_files,
        "custom_dataset": custom_dataset,
        "world_size": world_size,
        "rank": device.index,
    }
    load_next_files(**load_data_kwargs) 
    processed_batches = fast_forward_already_processed(num_of_processed_lines=num_of_processed_batches, **load_data_kwargs,)
    
    def next_data_batch():
        data = next(data_loader, None)
        if data is None:
            result = load_next_files(**load_data_kwargs)
            if not result:
                return None
            return next_data_batch()
        return data

    
    # Create a list to store the augmented records
    stats_dict = {}
    if os.path.exists(f"{augment_dir}/count_stats.csv"):
        with open(f"{augment_dir}/count_stats.csv", 'r') as f:
            generated_samples = int(f.readline().strip().split(",")[0])
    else:
        generated_samples = 0
    
    while generated_samples < GENERATION_TARGET:
        print(f"Processing batch {processed_batches}", flush=True)

        input_data = next_data_batch()
        if input_data is None:
            break
        annotated_data, current_stats = toolformer(**input_data)

        print(f"Gnerated {len(annotated_data)} samples!!!!", flush=True)
        for row in annotated_data[:50]:
            print(row['API_call_response_text'])
            print(row['loss_improvement'])
            print()

        save_stats(current_stats, stats_dict, augment_dir, device.index)

        save_generated_data(input_data, annotated_data, augment_dir, device.index)

        if os.path.exists(f"{augment_dir}/count_stats.csv"):
            with open(f"{augment_dir}/count_stats.csv", 'r') as f:
                generated_samples = int(f.readline().strip().split(",")[0])
        generated_samples += len(annotated_data)
        with open(f"{augment_dir}/count_stats.csv", 'w') as f:
            f.write(f"{generated_samples},{processed_batches}\n")

    
        logging.info(("$"*66+"\n")*3)
        logging.info(f"SO FAR, GNERATED {generated_samples} samples")
        print(f"So far, GNERATED {generated_samples} samples", flush=True)
        # Empty cuda cache
        torch.cuda.empty_cache()

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, full_dataset = False):
        self.data = data
        if full_dataset or all([sizes[i] == 1.0 for i in range(len(sizes))]):
            self.full_dataset = True
            self.partitions = [[x for x in range(0, len(data))]]
            return
        
        self.full_dataset = False
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        if self.full_dataset:
            partition = 0
        return Partition(self.data, self.partitions[partition])

def load_next_files(
    dataset_id: str,
    tool_name: str,
    data_batch_size: int = 128,
    skip_files: list[str] = [],
    custom_dataset: Dataset | None = None,
    world_size: int = 1,
    rank: int = 0,
):
    """
    Load the next batch of files from the dataset directory, or custom dataset.
    """
    global data_loader, file_batch
    tool_name = tool_name[0].lower() + tool_name[1:]

    if custom_dataset is not None:
        if data_loader is not None:
            return False
        data_loader = DataLoader(custom_dataset, batch_size=data_batch_size)
    else:
        file_list = [
            os.path.join(tool_name, file)
            for file in os.listdir(os.path.join("data", dataset_id, tool_name))
            if file.endswith(".csv") and file not in skip_files
        ]
        if file_batch >= len(file_list):
            return False

        dataset = DataPartitioner(load_dataset(dataset_id, data_files=file_list[file_batch : file_batch + FILE_BATCH_SIZE], split="train"), sizes=[1.0/world_size]*world_size).use(rank)

        file_batch += FILE_BATCH_SIZE
        data_loader = DataLoader(dataset, batch_size=data_batch_size)

    data_loader = iter(data_loader)
    return True

def fast_forward_already_processed(
        num_of_processed_lines: int = 0,
        **kwargs,
    ):
    global data_loader
    i = 0
    batch = 0
    while i < num_of_processed_lines:
        i += data_loader.batch_size
        batch += 1
        data = next(data_loader, None)

        if data is None:
            load_next_files(**kwargs)

    return batch

def save_stats(
    current_stats: dict,
    stats_dict: dict,
    augment_dir: str,
    rank: int = 0,
):
    stat_avgs = {}
    for key in current_stats:
        if key == 'Time key to example length for per example averaging': continue
        if key not in stats_dict:
            stats_dict[key] = []
        stats_dict[key].append(current_stats[key])

        total_examples_dict = current_stats['Time key to example length for per example averaging']
        if key.startswith("Time"):
            stat_avgs[f"{key} average (per batch)"] = sum(stats_dict[key])/len(stats_dict[key])
            print(f"{key} average (per batch): {stat_avgs[f'{key} average (per batch)']}", flush=True)
            logging.info(f"{key} average: {stat_avgs[f'{key} average (per batch)']}")
            if key in total_examples_dict and total_examples_dict[key] > 0:
                stat_avgs[f"{key} average (per example)"] = sum(stats_dict[key])/total_examples_dict[key]
                print(f"{key} average (per example): {stat_avgs[f'{key} average (per example)']}", flush=True)
                logging.info(f"{key} average (per example): {stat_avgs[f'{key} average (per example)']}")
        else:
            stat_avgs[key] = sum(stats_dict[key])/len(stats_dict[key])
            print(f"{key}: {stats_dict[key]}", flush=True)
            logging.info(f"{key}: {stats_dict[key]}")

    # Number of files in augment_dir that start with stats:
    add_header = not os.path.exists(f"{augment_dir}/stats/stats_{stats_num}.csv")
    with open(f"{augment_dir}/stats/stats_{stats_num}_{rank}.csv", "w") as f:
        writer = DictWriter(f, fieldnames=stats_dict.keys(), quoting=QUOTE_MINIMAL)
        if add_header:
            writer.writeheader()
        writer.writerow(stats_dict)
    with open(f"{augment_dir}/stats/stats_{stats_num}_{rank}_avgs.csv", "w") as f:
        writer = DictWriter(f, fieldnames=stat_avgs.keys(), quoting=QUOTE_MINIMAL)
        if add_header:
            writer.writeheader()
        writer.writerow(stat_avgs)

#CCNET_FIELD_NAMES = ['url', 'text', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket']
NEW_COLUMNS = ["API_calls_text", "API_call_response_text", "position",
               "loss_improvement", "arg_cohort", "raw_arg", "processed_arg"]

def save_generated_data(
    raw_data,
    annotated_data: list[dict],
    augment_dir: str,
    rank: int = 0,
):
    if len(annotated_data) == 0:
        return

    def file_path(file_counter):
        return f"{augment_dir}/augmented_data_{file_counter}_{rank}.csv"

    file_counter = len([file for file in os.listdir(augment_dir) if file.endswith('.csv') and not "stats" in file])
    if os.path.exists(file_path(file_counter)):
        with open(file_path(file_counter), "r") as f:
            file_lines = len(f.readlines())

        if file_lines > MAX_FILE_SIZE:
            file_counter += 1
            
    # DONT ELSE, IN CASE WE UP-ED THE COUNTER
    print(f"Raw data keys: {raw_data.keys()}", flush=True)
    print(f"Annotated data keys: {annotated_data[0].keys()}", flush=True)
    processed_data_cols = set(raw_data.keys()) | set(annotated_data[0].keys())
    processed_data_cols.discard("index")
    processed_data_cols = list(processed_data_cols)
    if not os.path.exists(file_path(file_counter)):
        with open(file_path(file_counter), 'w') as f:
            writer = DictWriter(f, fieldnames = processed_data_cols, quoting=QUOTE_ALL)
            writer.writeheader()

    augmented_rows = []
    for row in annotated_data:
        augmented_rows.append(data_row(row, raw_data, row['index']))
        print(f"Just appended {augmented_rows[-1]}", flush=True)

    with open(file_path(file_counter), 'a') as f:
        logging.info(f"Writing results to {file_path(file_counter)}")
        dictwriter_object = DictWriter(f, fieldnames=processed_data_cols, quoting=QUOTE_ALL)

        for row in augmented_rows:
            # Pass the dictionary as an argument to the Writerow()
            try:
                dictwriter_object.writerow(row)
            except Exception as e:
                print(e)
                print(row)
        
def data_row(processed_data, raw_data, index):
    new_row = {}
    for key in raw_data.keys():
        if key == "text": continue
        value = raw_data[key][index]
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        new_row[key] = value
    for key in NEW_COLUMNS + ["text"]:
        value = processed_data.get(key, None)
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        new_row[key] = value
    return new_row
