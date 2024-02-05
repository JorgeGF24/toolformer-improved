from csv import DictWriter, QUOTE_MINIMAL, QUOTE_ALL
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPTJConfig, LlamaConfig
from toolformer import Toolformer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset 
import os

from beartype import beartype

@beartype
def augment_db(
    tasks: list[dict],
    model_name:str="Mistral",
):
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    arg_gen_stoppers = []
    for k, v in tokenizer.get_vocab().items():
        if ']' in k or '→' in k or ')' in k:
            arg_gen_stoppers.append(v)
            
    # Task should have tf_kwargs with:
    # max_args_length, max_data_length, response_length, m_arg_samples, prompt_batch_size, filtering_batch_size, raw_tool_prompt, tool_name, tool, tool_check_duplicates, debug_level
    # Initialize the Toolformer
    toolformer = Toolformer(
        model=model,
        pad_token = tokenizer.pad_token,
        tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode,
        log_dir=task["augment_dir"] + "/logs",
        arg_gen_stoppers=torch.Tensor(arg_gen_stoppers),
        experiment_config=task,
        using_llama= model_name == "LLAMA",
    )
    
    # Load the dataset
    dataset = load_dataset("cc_news", split="train")
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create a list to store the augmented records
    augmented_records = []
    
    # Iterate over the dataset
    for batch in dataloader:
        # Generate the augmented records
        augmented_batch = toolformer.generate(
            batch["text"],
            tasks=tasks,
            max_length=100,
            num_return_sequences=1,
        )
        
        # Add the augmented records to the list
        augmented_records.extend(augmented_batch)
    
    # Write the augmented records to a CSV file
    with open("augmented_data.csv", "w") as f:
        writer = DictWriter(f, fieldnames=["original_text", "augmented_text"], quoting=QUOTE_MINIMAL)
        writer.writeheader()
        for record in augmented_records:
            writer.writerow(record)
    return augmented_records

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
                low_cpu_mem_usage=True, config=config, **cache_option)

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
                                                  **cache_option)
        
        tf_kwargs["api_start_token"] = "["

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
                                                  **cache_option)
        
        tf_kwargs["api_start_token"] = "["
    return tokenizer,model