from csv import QUOTE_ALL, DictWriter, QUOTE_MINIMAL
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPTJConfig, LlamaConfig
from toolformer import Toolformer
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset 
import os

from beartype import beartype


FILE_BATCH_SIZE = 1   # number of files to load at a time
GENERATION_TARGET = 50000                                                       
MAX_FILE_SIZE = 2000

file_batch = 0
stats_num = 0
data_loader = None

@beartype
def augment_db(
    dataset_dir,
    augment_dir,
    data_batch_size=1000,
    num_of_processed_batches=0,
    custom_dataset=None,
    skip_files:list=[],
    model_name:str="GPTJ",  # "GPTJ" or "LLAMA",
    experiment_config:dict={},
    generated_so_far=0,
    **kwargs,
):
    global data_loader, file_batch, stats_num
    previous_model = None

    file_batch = 0
    stats_num = len([file for file in os.listdir(os.path.join(augment_dir,"stats")) if file.startswith("stats")])
    os.makedirs(augment_dir + "/stats", exist_ok=True)
    
    if previous_model != model_name:
        tokenizer, model = load_model_and_tokenizer(model_name)

        arg_gen_stoppers = []
        for k, v in tokenizer.get_vocab().items():
            if ']' in k or '→' in k or ')' in k:
                arg_gen_stoppers.append(v)

    previous_model = model_name
            
    # Task should have tf_kwargs with:
    # max_args_length, max_data_length, response_length, m_arg_samples, prompt_batch_size, filtering_batch_size, raw_tool_prompt, tool_name, tool, tool_check_duplicates, debug_level
    # Initialize the Toolformer
    toolformer = Toolformer(
        model=model,
        pad_token = tokenizer.pad_token,
        tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode,
        log_dir=augment_dir + "/logs",
        arg_gen_stoppers=torch.Tensor(arg_gen_stoppers),
        experiment_config=experiment_config,
        using_llama= model_name == "LLAMA",
        **kwargs
    ).cuda()
    
    load_data_kwargs = {
        "dataset_dir": dataset_dir,
        "data_batch_size": data_batch_size,
        "skip_files": skip_files,
        "custom_dataset": custom_dataset,
    }
    data_loader = load_next_files(**load_data_kwargs) 
    processed_batches = fast_forward_already_processed(num_of_processed_lines=num_of_processed_batches, **load_data_kwargs,)
    
    def next_data_batch():
        data = next(data_loader, None)
        if data is None:
            load_next_files(dataset_dir, data_batch_size, skip_files, custom_dataset)
            return next_data_batch()
        return data

    
    # Create a list to store the augmented records
    stats_dict = {}
    generated_samples = generated_so_far
    
    while generated_samples < GENERATION_TARGET:
        print(f"Processing batch {processed_batches}", flush=True)

        input_data = next_data_batch()
        annotated_data, current_stats = toolformer(**input_data)

        print(f"Gnerated {len(annotated_data)} samples!!!!", flush=True)
        for row in annotated_data[:50]:
            print(row['API_call_response_text'])
            print(row['loss_improvement'])
            print()

        save_stats(current_stats, stats_dict, augment_dir)

        save_generated_data(input_data, annotated_data, augment_dir)
        generated_samples += len(annotated_data)

        with open(f"{augment_dir}/count_stats.csv", 'w') as f:
            f.write(f"{generated_samples},{processed_batches}\n")

    
        logging.info(("$"*66+"\n")*3)
        logging.info(f"SO FAR, GNERATED {generated_samples} samples")
        print(f"So far, GNERATED {generated_samples} samples", flush=True)
        # Empty cuda cache
        torch.cuda.empty_cache()


def load_model_and_tokenizer(model_name):
    cache_dir = None
    cache_option = {} if cache_dir is None else {"cache_dir": cache_dir}
    
    if model_name == "GPTJ":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, **cache_option)
        tokenizer.pad_token=tokenizer.eos_token
        config = GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B", padding_idx=tokenizer.pad_token_id)

        model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True, config=config, **cache_option).cuda()

    elif model_name == "LLAMA":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                   token="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",
                                                   **cache_option)

        tokenizer.add_bos_token = False
        
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                             token="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",
                                             padding_idx=tokenizer.pad_token_id)
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                  token="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",
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True,
                                                  config=config,
                                                  **cache_option).cuda()

    elif model_name == "LLAMA-big":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf",
                                                   token="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",
                                                   **cache_option)

        tokenizer.add_bos_token = False

        #tokenizer.add_tokens(["[PAD]"])
        
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-70b-hf", 
                                             token="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",
                                             padding_idx=tokenizer.pad_token_id)
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf",
                                                  token="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",
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True,
                                                  config=config,
                                                  **cache_option).cuda()
    elif model_name == "MISTRAL":

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", **cache_option)
        tokenizer.pad_token=tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True, **cache_option).cuda()

    return tokenizer,model

def load_next_files(
    dataset_dir: str,
    data_batch_size: int = 128,
    skip_files: list[str] = [],
    custom_dataset: Dataset | None = None,
):
    """
    Load the next batch of files from the dataset directory, or custom dataset.
    """
    global data_loader
    if custom_dataset is not None:
        data_loader = DataLoader(custom_dataset, batch_size=data_batch_size)
    else:
        file_list = [
            file
            for file in os.listdir(dataset_dir)
            if file.endswith(".csv") and file not in skip_files
        ]
        file_batch = file_batch
        dataset = load_dataset(dataset_dir, data_files=file_list[file_batch : file_batch + FILE_BATCH_SIZE])

        file_batch += FILE_BATCH_SIZE
        data_loader = DataLoader(dataset, batch_size=data_batch_size)

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
            print(f"{key} average (per batch): {stat_avgs[f"{key} average (per batch)"]}", flush=True)
            logging.info(f"{key} average: {stat_avgs[f"{key} average (per batch)"]}")
            if key in total_examples_dict and total_examples_dict[key] > 0:
                stat_avgs[f"{key} average (per example)"] = sum(stats_dict[key])/total_examples_dict[key]
                print(f"{key} average (per example): {stat_avgs[f"{key} average (per example)"]}", flush=True)
                logging.info(f"{key} average (per example): {stat_avgs[f"{key} average (per example)"]}")
        else:
            stat_avgs[key] = sum(stats_dict[key])/len(stats_dict[key])
            print(f"{key}: {stats_dict[key]}", flush=True)
            logging.info(f"{key}: {stats_dict[key]}")

    # Number of files in augment_dir that start with stats:
    add_header = not os.path.exists(f"{augment_dir}/stats/stats_{stats_num}.csv")
    with open(f"{augment_dir}/stats/stats_{stats_num}.csv", "w") as f:
        writer = DictWriter(f, fieldnames=stats_dict.keys(), quoting=QUOTE_MINIMAL)
        if add_header:
            writer.writeheader()
        writer.writerow(stats_dict)
    with open(f"{augment_dir}/stats/stats_{stats_num}_avgs.csv", "w") as f:
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
):
    if len(annotated_data) == 0:
        return

    def file_path(file_counter):
        return f"{augment_dir}/augmented_data_{file_counter}.csv"

    file_counter = len([file for file in os.listdir(augment_dir) if file.endswith('.csv') and not "stats" in file]) - 1
    if os.path.exists(file_path(file_counter)):
        with open(file_path(file_counter), "r") as f:
            file_lines = len(f.readlines())

        if file_lines > MAX_FILE_SIZE:
            file_counter += 1
            
    # DONT ELSE, IN CASE WE UP-ED THE COUNTER
    processed_data_cols = raw_data.keys() + annotated_data[0].keys()
    if not os.path.exists(file_path(file_counter)):
        with open(file_path(file_counter), 'w') as f:
            writer = DictWriter(f, fieldnames = processed_data_cols, quoting=QUOTE_ALL)
            writer.writeheader()

    augmented_rows = []
    for row in annotated_data:
        augmented_rows.append(data_row(row, raw_data, row['index']))

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