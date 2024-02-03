from csv import DictWriter, QUOTE_MINIMAL
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPTJConfig, LlamaConfig
from toolformer import Toolformer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import csv

def compare_probs(
    tool_name,
    tool,
    tool_check_duplicates,
    raw_tool_prompt,
    dataset_dir,
    augment_dir,
    max_args_length,
    max_data_length,
    prompt_batch_size,
    filtering_batch_size,
    extra_data_columns,
    response_length=15,
    data_batch_size=1000,
    m_arg_samples=5,
    start_at_batch=0,
    custom_dataset=None,
    skip_files:list=[],
    model_name:str="GPTJ",  # "GPTJ" or "LLAMA",
    **kwargs,
):
    # Empty cuda cache
    torch.cuda.empty_cache()

    cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
    cache_option = {"cache_dir": cache_dir} if cache_dir else {} 



    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, **cache_option)

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



    # toolformer
    toolformer = Toolformer(
        model=model,
        pad_token = tokenizer.pad_token,
        max_arg_length=max_args_length,
        max_data_length=max_data_length,
        max_response_length=response_length,
        m_arg_samples=m_arg_samples,
        prompt_batch_size=prompt_batch_size,
        filtering_batch_size=filtering_batch_size,
        raw_tool_prompt=raw_tool_prompt,
        tool_name=tool_name,
        tool=tool,
        tool_check_duplicates=tool_check_duplicates,
        tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode,
        log_dir=augment_dir + "/logs",
        using_llama= model_name == "LLAMA",
        **kwargs
    )
    print("Toolformer created")


    # Read data from the dataset
    # Store csv files in Dataset object:

    file_batch = 0
    file_batch_size = 1   # number of files to load at a time
    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')and file not in skip_files]

    if custom_dataset:
        dataloader = DataLoader(custom_dataset, batch_size=data_batch_size)
        data_iter = iter(dataloader)
    else:
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
        "GPTJ_values": [],
        "GPTJ_lengths": [],
        #"LLAMA_values": [],
        #"LLAMA_lengths": [],
    }

    i = 0
    while i < 10:
        print(f"Processing batch {i}", flush=True)
        print("LOL", flush=True)
        i += 1

        values, lengths = toolformer.compare_pos_sampling(data=data['text'], model=model, encode=tokenizer.encode)

        stats_dict["GPTJ_values"].extend(values)
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






    """
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
    model.resize_token_embeddings(len(tokenizer))

    # toolformer
    toolformer = Toolformer(
        model=model,
        pad_token = tokenizer.pad_token,
        max_arg_length=max_args_length,
        max_data_length=max_data_length,
        max_response_length=response_length,
        m_arg_samples=m_arg_samples,
        prompt_batch_size=prompt_batch_size,
        filtering_batch_size=filtering_batch_size,
        raw_tool_prompt=raw_tool_prompt,
        tool_name=tool_name,
        tool=tool,
        tool_check_duplicates=tool_check_duplicates,
        tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode,
        log_dir=augment_dir + "/logs",
        using_llama= model_name == "LLAMA",
        **kwargs
    )
    print("Toolformer created")

    torch.cuda.empty_cache()
    file_batch = 0
    if custom_dataset:
        dataloader = DataLoader(custom_dataset, batch_size=data_batch_size)
        data_iter = iter(dataloader)
    else:
        print(f"Loading {start_at_batch}th batch of the dataset")
        dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[:file_batch_size], split="train")
        #dataset.set_format("torch")
        dataloader = DataLoader(dataset, batch_size=data_batch_size)
        data_iter = iter(dataloader)

    i = 0
    while i < 10:
        print(f"Processing batch {i}", flush=True)
        i += 1

        values, lengths = toolformer.compare_pos_sampling(data=data['text'], model=model, encode=tokenizer.encode)

        stats_dict["GPTJ_values"].extend(values)
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
            break"""

    # Print average of each list in dict:
    for key, value in stats_dict.items():
        print("Average of list")
        print(f"{key}: {sum(value)/len(value)}")
        print("Standard deviation of list")
        print(f"{key}: {torch.std(torch.tensor(value))}")
        print("Max of list")
        print(f"{key}: {max(value)}")
        print("Min of list")
        print(f"{key}: {min(value)}")

    # Save stats to a csv file that has "GPTJ_values" and "LLAMA_values" as column headers and the values as rows
    with open(f"{augment_dir}/position_probs.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["GPTJ_values"])#, "LLAMA_values"])
        writer.writerows(zip(stats_dict["GPTJ_values"]))#, stats_dict["LLAMA_values"]))
