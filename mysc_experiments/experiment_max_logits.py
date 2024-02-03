from csv import DictWriter, QUOTE_MINIMAL
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPTJConfig, LlamaConfig
from toolformer import Toolformer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import csv
import statistics


from functools import partial
from torch.nn.utils.rnn import pad_sequence

pad_sequence = partial(pad_sequence, batch_first=True)

@torch.no_grad()
def compare_max_logits(
    dataset_dir,
    augment_dir,
    max_data_length,
    data_batch_size=1000,
    start_at_batch=0,
    skip_files:list=[],
    **kwargs,
):
    # Empty cuda cache
    torch.cuda.empty_cache()

    cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
    cache_option = {"cache_dir": cache_dir} if cache_dir else {}

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", **cache_option)

    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token="[PAD]"

    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, **cache_option
    ).cuda()    
    model.resize_token_embeddings(len(tokenizer))

    model.eval()



    # Read data from the dataset
    # Store csv files in Dataset object:

    file_batch = 0
    file_batch_size = 1   # number of files to load at a time
    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')and file not in skip_files]

    print(f"Loading {start_at_batch}th batch of the dataset")
    dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[:file_batch_size], split="train")
    #dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=data_batch_size)
    data_iter = iter(dataloader)

    print("Loaded dataset", flush=True)
    
    data = next(data_iter, None)

    i = 0
    while i < start_at_batch:
        i += 1
        data = next(data_iter, None)

        if data is None:
            file_batch += 1
            if file_batch*file_batch_size < len(file_list):
                print(f"Loading {file_batch}th batch of the dataset")
                dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[file_batch * file_batch_size:(file_batch + 1) * file_batch_size], split="train")
                #dataset.set_format("torch")
                dataloader = DataLoader(dataset, batch_size=data_batch_size)
                data_iter = iter(dataloader)
                data = next(data_iter, None)

    stats_dict = {
        "GPTJ_probs": [],
        "GPTJ_logits": [],
        "GPTJ_lengths": [],
        "LLAMA_logits": [],
        "LLAMA_probs": [],
        "LLAMA_lengths": [],
    }

    i = 0
    while i*data_batch_size < 10000:
        print(f"Processing batch {i} ({i*data_batch_size/100}%)", flush=True)

        tokenized_text = [tokenizer(text, return_tensors="pt").input_ids.view(-1) for text in data['text']]
    
        tokenized_text = sorted(tokenized_text, key=lambda x: x.shape[0], reverse=True)
        lengths = [text.shape[0] for text in tokenized_text]
        torch_lens = torch.tensor(lengths)

        if False:
            input = pad_sequence(tokenized_text, padding_value=tokenizer.pad_token_id).cuda().long()
            
            # Arange until input.shape[1] repeated batch_size times with output shape (batch_size, input.shape[1])
            mask = torch.arange(input.shape[1]).unsqueeze(0) >= torch_lens.unsqueeze(1)
            
            logits = model(input, use_cache=False).logits.masked_fill(mask.unsqueeze(2).cuda(), 0)
            max_logits, _ = logits.max(dim=-1)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs, _ = probs.max(dim=-1)


            stats_dict["GPTJ_logits"].extend(max_logits.view(-1).tolist())        
            stats_dict["GPTJ_probs"].extend(max_probs.view(-1).tolist())
        stats_dict["GPTJ_lengths"].extend(lengths)

        # Empty cuda cache
        torch.cuda.empty_cache()

        data = next(data_iter, None)
        if data is None:
            file_batch += 1
            if file_batch*file_batch_size < len(file_list):
                print(f"Loading {file_batch}th batch of the dataset")
                dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[file_batch * file_batch_size:(file_batch + 1) * file_batch_size], split="train")
                #dataset.set_format("torch")
                dataloader = DataLoader(dataset, batch_size=data_batch_size)
                data_iter = iter(dataloader)
                data = next(data_iter, None)
                continue
            break


        i += 1

    
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                token="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",
                                                **cache_option)

    tokenizer.add_bos_token = False

    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token="[PAD]"

    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                token="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",
                                                padding_idx=tokenizer.pad_token_id)
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                token="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",
                                                config=config,
                                                **cache_option).cuda()

    model.resize_token_embeddings(len(tokenizer))
    kwargs["api_start_token"] = "["





    model.eval()


    torch.cuda.empty_cache()
    file_batch = 0

    print(f"Loading {start_at_batch}th batch of the dataset")
    dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[:file_batch_size], split="train")
    #dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=data_batch_size)
    data_iter = iter(dataloader)

    i = 0
    while i*data_batch_size < 10000:
        print(f"Processing batch {i} ({i*data_batch_size/100}%)", flush=True)

        tokenized_text = [tokenizer(text, return_tensors="pt").input_ids.view(-1) for text in data['text']]
    
        tokenized_text = sorted(tokenized_text, key=lambda x: x.shape[0], reverse=True)
        lengths = [text.shape[0] for text in tokenized_text]

        if False:
            torch_lens = torch.tensor(lengths)
            input = pad_sequence(tokenized_text, padding_value=tokenizer.pad_token_id).cuda().long()
            
            # Arange until input.shape[1] repeated batch_size times with output shape (batch_size, input.shape[1])
            mask = torch.arange(input.shape[1]).unsqueeze(0) >= torch_lens.unsqueeze(1)
            
            logits = model(input, use_cache=False).logits.masked_fill(mask.unsqueeze(2).cuda(), 0)
            max_logits, _ = logits.max(dim=-1)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs, _ = probs.max(dim=-1)


            stats_dict["LLAMA_logits"].extend(max_logits.view(-1).tolist())        
            stats_dict["LLAMA_probs"].extend(max_probs.view(-1).tolist())
        stats_dict["LLAMA_lengths"].extend(lengths)

        # Empty cuda cache
        torch.cuda.empty_cache()

        data = next(data_iter, None)
        if data is None:
            file_batch += 1
            if file_batch*file_batch_size < len(file_list):
                print(f"Loading {file_batch}th batch of the dataset")
                dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[file_batch * file_batch_size:(file_batch + 1) * file_batch_size], split="train")
                #dataset.set_format("torch")
                dataloader = DataLoader(dataset, batch_size=data_batch_size)
                data_iter = iter(dataloader)
                data = next(data_iter, None)
                continue
            break


        i += 1

    # Print average of each list in dict:
    for key, value in stats_dict.items():
        if len(value) == 0:
            continue
        print("Average of list")
        print(f"{key}: {sum(value)/len(value)}")
        print("Standard deviation of list")
        print(f"{key}: {statistics.stdev(value)}")
        print("Max of list")
        print(f"{key}: {max(value)}")
        print("Min of list")
        print(f"{key}: {min(value)}")

    # Save stats to a csv file that has "GPTJ_values" and "LLAMA_values" as column headers and the values as rows
    with open(f"{augment_dir}/max_logits_llama.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Max logits", "Max probs"])
        writer.writerows(zip(stats_dict["LLAMA_logits"], stats_dict["LLAMA_probs"]))

    # Save stats to a csv file that has "GPTJ_values" and "LLAMA_values" as column headers and the values as rows
    with open(f"{augment_dir}/max_logits_gptj.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Max logits", "Max probs"])
        writer.writerows(zip(stats_dict["GPTJ_logits"], stats_dict["GPTJ_probs"]))
