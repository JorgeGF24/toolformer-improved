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
def augment_database(
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
    new_data_columns = ["API_calls_text", "API_call_response_text", "position", "loss_improvement"],
    num_of_processed_batches=0,
    custom_dataset=None,
    erase=False,
    skip_files:list=[],
    model_name:str="GPTJ",  # "GPTJ" or "LLAMA",
    debug_level:int=1,
    experiment_config:dict={},
    generated_so_far=0,
    **kwargs,
):
    # Empty cuda cache
    torch.cuda.empty_cache()

    print(f"ERASE IS SET TO {erase}")
    print("Loaded dataset", flush=True)
    
    old_field_names = ['url', 'text', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket'] # in database, but will add 'API_calls_text', 'API_call_response_text', 'loss_improvement', 
    old_field_names = old_field_names[:2] + extra_data_columns + old_field_names[2:]
    new_field_names = old_field_names[:2] + new_data_columns + old_field_names[2:]
    generation_target = 50000                                                       
    max_file_size = 2000
    
    def file_name(id):
        return f"{augment_dir}/{id}.csv"
    file_counter = len([file for file in os.listdir(augment_dir) if file.endswith('.csv') and not file.endswith("stats.csv")]) - 1
    file_lines = 0

    # Check if file_name(file_counter) exists and count number of lines:
    if os.path.exists(file_name(file_counter)):
        with open(file_name(file_counter), "r") as f:
            file_lines += len(f.readlines())
    
    if file_lines > max_file_size:
        file_counter += 1
        file_lines = 0
    
    def data_row(processed_data, raw_data, index):
        new_row = {}
        for key in old_field_names:
            if key == "text": continue
            value = raw_data[key][index]
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.tolist()
            new_row[key] = value
        for key in new_data_columns+["text"]:
            value = processed_data.get(key, None)
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.tolist()
            new_row[key] = value
        return new_row

    # Check if the augment_dir exists and create it recursively if not

    if not os.path.exists(augment_dir):
        os.makedirs(augment_dir)
    os.makedirs(augment_dir + "/stats", exist_ok=True)

    # check if the file calendar_0.csv exists and create them if not with field_names as the header
    prepare_new_file(file_name(file_counter), new_field_names, erase)
    
    generated_samples = generated_so_far


    stats_dict = {}

    while generated_samples < generation_target:

        annotated_data, current_stats = toolformer(*args)


        """        augmented_rows = []
        for row in annotated_data:
            augmented_rows.append(data_row(row, data, row['index']))

        if len(annotated_data) > 0:
            with open(file_name(file_counter), 'a') as f:
                logging.info(f"Writing results to {file_name(file_counter)}")
                dictwriter_object = DictWriter(f, fieldnames=new_field_names, quoting=QUOTE_ALL)

                for row in augmented_rows:
                    # Pass the dictionary as an argument to the Writerow()
                    try:
                        dictwriter_object.writerow(row)
                    except Exception as e:
                        print(e)
                        print(row)
                
                generated_samples += len(annotated_data)
                file_lines += len(annotated_data)
                
            if file_lines > max_file_size: #
                file_counter += 1
                file_lines = 0
                
                prepare_new_file(file_name(file_counter), new_field_names, erase)"""

        # Write a count_stats.csv file with the number of generated samples and processed batches:
        with open(f"{augment_dir}/count_stats.csv", 'w') as f:
            f.write(f"{generated_samples},{i}\n")

        for key in current_stats:
            if key == 'Time key to example length for per example averaging': continue
            if key not in stats_dict:
                stats_dict[key] = []
            stats_dict[key].append(current_stats[key])

            total_examples_dict = current_stats['Time key to example length for per example averaging']
            if key.startswith("Time"):
                print(f"{key} average (per batch): {sum(stats_dict[key])/len(stats_dict[key])}", flush=True)
                logging.info(f"{key} average: {sum(stats_dict[key])/len(stats_dict[key])}")
                if key in total_examples_dict and total_examples_dict[key] > 0:
                    print(f"{key} average (per example): {sum(stats_dict[key])/total_examples_dict[key]}", flush=True)
                    logging.info(f"{key} average (per example): {sum(stats_dict[key])/total_examples_dict[key]}")
            else:
                print(f"{key}: {stats_dict[key]}", flush=True)
                logging.info(f"{key}: {stats_dict[key]}")

        logging.info(("$"*66+"\n")*3)
        logging.info(f"SO FAR, GNERATED {generated_samples} samples")
        print(f"So far, GNERATED {generated_samples} samples", flush=True)
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


    print(f"Finished augmenting database. Generated {generated_samples} samples.")
    if data is None:
        print("Reached end of database.")
    else:  
        print("Generated target number of samples.")

@beartype
def prepare_new_file(file_name:str, header:list, erase:bool):
    if not os.path.exists(file_name) or erase:
        with open(file_name, 'w') as f:
            writer = DictWriter(f, fieldnames=header)
            writer.writeheader()