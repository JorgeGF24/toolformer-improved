import csv
import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda import device_count

from torch.nn.parallel import DistributedDataParallel as DDP

from toolformer import augment_db
from mysc_experiments.experiment_pos_thresholding import compare_probs
from mysc_experiments.experiment_max_logits import compare_max_logits

from beartype import beartype
from beartype.typing import List

from tools import Calendar, Calculator, WikiSearch, calc_parse, init_wikisearch, wiki_parse


def identity_check(arg):
    return list(range(len(arg)))


prompt = f"""These are examples where we use results from a calendar tool that returns the current date to complete the sentence. The calls should help you get information required to complete the text, such as the temporal context of a person, action or general information. You can call the API by writing "[Calendar()]". Here are some examples of API calls:

Example 1: Today is the first [Calendar()→ Today is Friday, 01/01/2019] Friday of the year.

Example 2: The president of the United States is [Calendar()→ Today is Tuesday, 11/02/2007] George W. Bush.

Example 3: [PAD]"""


# Bear type check
@beartype
def calculator_check(arg: List[str]) -> List[int]:
    clean_arg = [x.replace(" ", "") for x in arg]
    valid_indices = [0]
    for i in range(1, len(clean_arg)):
        if clean_arg[i] not in clean_arg[:i]:
            valid_indices.append(i)
    return valid_indices


calculator_prompt = f"""Your task is to add calls to a Calculator API to a piece of text. The calls should help you get information required to complete the text. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of API calls:

Input: Last year we collected 237342 apples, double of what we collected this year: 118671.
Output: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→118671] 118671.

Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18+12*3)→54] 54.

Input: A total of 252 matches were played, and 723 goals were scored (an average of 2.87 per match). This is twenty goals more than the 703 goals last year.
Output: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723/252)→2.87] 2.87 per match). This is twenty goals more than the [Calculator(723-20)→703] 703 goals last year.

Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011-1994)→17] 17 years.

Input: [PAD]
Output: """

calculator_arg_prompt = f"""These are examples where we use results from a calculator tool to complete the sentence. The calls should help you get information required to complete the text. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of API calls:

Example 1: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→118671] 118671.

Example 2: The number in the next term is 18 + 12 x 3 = [Calculator(18+12*3)→54] 54.

Example 3: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723/252)→2.87] 2.87 per match). This is twenty goals more than in 2013, when the total was [Calculator(723-20)→703] 703 goals last year.

Example 4: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011-1994)→17] 17 years.

Example 5: """

searcher = None
stopwords = None


@beartype
def simple_check(args: List[str]):
    # Order the words in the list of strings in alphabetical order
    for i, arg in enumerate(args):
        args[i] = ' '.join(sorted(arg.split()))

    valid_indices = [0]
    for i in range(1, len(args)):
        if args[i] not in args[:i]:
            valid_indices.append(i)

    # Check if first element is empty
    if len(args[0]) == 0:
        valid_indices.pop(0)
    return valid_indices


wikipedia_search_prompt = f"""Your task is to complete a given piece of text. You can use a Wikipedia Search API to look up information. You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up. Here are some examples of API calls:

Input: The colors on the flag of Ghana have the following meanings: red is for the blood of martyrs, green for forests, and gold for mineral wealth.
Output: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch(Ghana flag red meaning)] the blood of martyrs, green for forests, and gold for mineral wealth.

Input: But what are the risks during production of nanomaterials? Some nanomaterials may give rise to various kinds of lung damage.
Output: But what are the risks during production of nanomaterials? [WikiSearch(nanomaterial production risks)] Some nanomaterials may give rise to various kinds of lung damage.

Input: Metformin is the first-line drug for patients with type 2 diabetes and obesity.
Output: Metformin is the first-line drug for [WikiSearch(Metformin first-line drug)] patients with type 2 diabetes and obesity.

Input: [PAD]
Output: """

wikipedia_arg_prompt = f"""These are examples where we use results from a Wikipedia search to complete the sentence. You can use a Wikipedia Search API to look up information. You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up. Here are some examples of API calls:

Example 1: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch(Ghana flag red meaning)] the blood of martyrs, green for forests, and gold for mineral wealth.

Example 2: But what are the risks during production of nanomaterials? [WikiSearch(nanomaterial production risks)] Some nanomaterials may give rise to various kinds of lung damage.

Example 3: Metformin is the first-line drug for [WikiSearch(Metformin first-line drug)] patients with type 2 diabetes and obesity.

Example 4: """

new_columns = ["API_calls_text", "API_call_response_text", "position",
               "loss_improvement", "arg_cohort", "raw_arg", "processed_arg"]


# 30 args len, 50 data len, 5 k arg samples, 14 prompt batch size, 50 filtering batch size FITS in GPUC

def augment_calend(config):
    print("Augmenting calendar data...")

    config['augment_dir'] = config['augment_dir'] + "_" + config['model_name']
    # Check if there is a count_stats.csv file in the augment_dir
    if os.path.exists(os.path.join(config['augment_dir'], "count_stats.csv")):
        with open(os.path.join(config['augment_dir'], "count_stats.csv"), "r") as f:
            reader = csv.reader(f)
            count_stats = list(reader)
            config["generated_so_far"] = int(count_stats[0][0])
            config["num_of_processed_batches"] = int(count_stats[0][1])
    print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for key, values in config.items():
        print(f"{key}: {values}")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    augment_db(
        experiment_config=config,
        **config
    )


def augment_calc(config):
    print("Augmenting calculator data...")
    config['augment_dir'] = config['augment_dir'] + "_" + config['model_name']
    # Check if there is a count_stats.csv file in the augment_dir
    if os.path.exists(os.path.join(config['augment_dir'], "count_stats.csv")):
        with open(os.path.join(config['augment_dir'], "count_stats.csv"), "r") as f:
            reader = csv.reader(f)
            count_stats = list(reader)
            config["generated_so_far"] = int(count_stats[0][0])
            config["num_of_processed_batches"] = int(count_stats[0][1])
    print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for key, values in config.items():
        print(f"{key}: {values}")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    augment_db(
        experiment_config=config,
        **config
    )


def augment_wiki(config):
    print("Augmenting wikipedia data...", flush=True)
    init_wikisearch()
    config['augment_dir'] = config['augment_dir'] + "_" + config['model_name']
    # Check if there is a count_stats.csv file in the augment_dir
    if os.path.exists(os.path.join(config['augment_dir'], "count_stats.csv")):
        with open(os.path.join(config['augment_dir'], "count_stats.csv"), "r") as f:
            reader = csv.reader(f)
            count_stats = list(reader)
            config["generated_so_far"] = int(count_stats[0][0])
            config["num_of_processed_batches"] = int(count_stats[0][1])

    print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for key, values in config.items():
        print(f"{key}: {values}")
    augment_db(
        experiment_config=config,
        **config
    )

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def experiment(rank, name, world_size):

    setup(rank, world_size)

    size_factor = 0.5 if len(sys.argv) > 1 and sys.argv[1] == "24" else 1

    print(f"Starting experiment {name}....", flush=True)
    print("Wish me luck!")

    data = [{'text': "His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of 77.7.", "date": "2011-12-01"},
            {'text': "Incredibly fast in shallow water, and the deepest shade of blue, the twaite shad are one of the rarest fish species.", "date": "2011-12-01"},
            {'text': "Once upon a time there were 11453626 cows in my farm, but half of them died, so now there are 5726813 cows.", "date": "2011-12-01"},
            {'text': "Once upon a time there were 11453626 cows in my farm, but half of them died, so now there are 5726813 cows. Welcome welcome welcole and hew ajhdsw fhwe fedkjq feku qwfehwdf wef wdf wef sf wqf reg efdre wfqw eqw efwqe fer gvrt er fdf w efeff", "date": "2011-12-01"}, ]

    calen_default_config = {
        'tool_name': "Calendar",
        'tool': Calendar,
        'tool_check_duplicates': identity_check,
        'raw_tool_prompt': prompt,
        'dataset_dir': "/vol/bitbucket/jg2619/data/preprocessed/big_load/calendar",
        'augment_dir': "/vol/bitbucket/jg2619/data/augmented2/calendar3",
        'max_args_length': 0,
        'max_data_length': 256,
        'k_positions': 8,
        'm_arg_samples': 1,
        'prompt_batch_size': 104,
        'filtering_batch_size': 2500,
        'extra_data_columns': ['date'],
        'new_data_columns': new_columns,
        'filter_threshold': 0.2,
        'device': rank if device_count() > 0 else "cpu",
        'world_size': world_size
    }

    wiki_default_config = {
        'tool_name': 'WikiSearch',
        'tool': WikiSearch,
        'tool_check_duplicates': simple_check,
        'preprocess_args': wiki_parse,
        'raw_tool_prompt': wikipedia_search_prompt,
        'max_args_length': 12,
        'max_data_length': 150,
        'response_length': 75,
        'extra_data_columns': [],
        'new_data_columns': new_columns,
        'sampling_temperature': 0.5,
        'filter_threshold': 0.2,
        'device': rank if device_count() > 0 else "cpu",
        'world_size': world_size
    }

    calc_default_config = {
        'tool_name': "Calculator",
        'tool': Calculator,
        'tool_check_duplicates': calculator_check,
        'preprocess_args': calc_parse,
        'raw_tool_prompt': calculator_prompt,
        'max_args_length': 30,
        'max_data_length': 180,
        'extra_data_columns': [],
        'new_data_columns': new_columns,
        'sampling_temperature': 0.5,
        'filter_threshold': 0.5,
        'device': rank if device_count() > 0 else "cpu",
        'world_size': world_size
    }

    name = name.replace("_", " ")
    match name:
        case "Comparing probabilities":
            print("Comparing probabilities...", flush=True)
            config = wiki_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/wikiSearch',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer-luci/compare_probs',
                'm_arg_samples': 5,
                'prompt_batch_size': 64,
                'filtering_batch_size': 512,
                # 'raw_arg_prompt': wikipedia_arg_prompt,
                'erase': True,
                # 'skip_files': ["0.csv"],
                # 'num_of_processed_batches':3,
                'model_name': "GPTJ",    # GPTJ or LLAMA
                'debug_level': 2,
                # 'custom_dataset': data,
                # 'data_batch_size': 14
            }

            print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for key, values in config.items():
                print(f"{key}: {values}")

            compare_probs(**config)

            raise Exception("Finished testing pipeline")

        case "Comparing max logits":
            print("Comparing max logits...", flush=True)
            config = {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/calculator',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer-luci/compare_lengthss',
                'max_data_length': wiki_default_config['max_data_length'],
                'data_batch_size': 200
            }

            print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for key, values in config.items():
                print(f"{key}: {values}")

            compare_max_logits(**config)

            raise Exception("Finished testing pipeline")

        case "Lightweight pipeline test":
            print("TESTING PIPELINE STANDARD arg prompt...", flush=True)
            config = wiki_default_config | {
                'dataset_dir': '/home/jorge/ccnet/merged/processed/wikiSearch',
                'augment_dir': '/home/jorge/toolformer-improved/data/debug',
                'm_arg_samples': 5,
                'prompt_batch_size': int(202*size_factor),
                'filtering_batch_size': int(6000*size_factor),
                # 'raw_arg_prompt': wikipedia_arg_prompt,
                # 'skip_files': ["0.csv"],
                # 'num_of_processed_batches':3,
                'model_name': "MOCK",    # GPTJ or LLAMA
                'debug_level': 2,
                'custom_dataset': data,
                # 'data_batch_size': 14
            }
            augment_wiki(config)
            print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for key, values in config.items():
                print(f"{key}: {values}")

        case "Lightweight pipeline test trick":
            print("TESTING PIPELINE STANDARD arg prompt...", flush=True)
            config = wiki_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/wikiSearch',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer-luci/debugging',
                'm_arg_samples': 5,
                'prompt_batch_size': int(202*size_factor),
                'filtering_batch_size': int(6000*size_factor),
                'raw_arg_prompt': wikipedia_arg_prompt,
                'erase': True,
                # 'skip_files': ["0.csv"],
                # 'num_of_processed_batches':3,
                'model_name': "LLAMA",    # GPTJ or LLAMA
                'debug_level': 2,
                'custom_dataset': data,
                # 'data_batch_size': 14
            }
            augment_wiki(config)
            print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for key, values in config.items():
                print(f"{key}: {values}")

        case "Lightweight pipeline test CALENDAR":
            print("TESTING CALENDAR PIPELINE STANDARD arg prompt...", flush=True)
            config = calen_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/wikiSearch',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer-luci/debugging',
                'm_arg_samples': 1,
                'prompt_batch_size': int(202*size_factor),
                'filtering_batch_size': int(6000*size_factor),
                # 'raw_arg_prompt': wikipedia_arg_prompt,
                'erase': True,
                # 'skip_files': ["0.csv"],
                # 'num_of_processed_batches':3,
                'model_name': "LLAMA",    # GPTJ or LLAMA
                'debug_level': 2,
                'custom_dataset': data,
                "extra_data_columns": ["date"],
                # 'data_batch_size': 14
            }
            augment_wiki(config)
            print("\n CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for key, values in config.items():
                print(f"{key}: {values}")

        case "Experiment 1":
            print(
                "Augmenting wikipedia data with SEPARATE arg prompt and 0.9 temperature...", flush=True)
            config = wiki_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_reverse/wikiSearch',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_standard/wikiSearch',
                'm_arg_samples': 8,
                'prompt_batch_size': 64,
                'filtering_batch_size': 256,
                'raw_arg_prompt': wikipedia_arg_prompt,
            }
            augment_wiki(config)

        case "LLAMA Calculator":
            print("Augmenting calculator data with STANDARD arg prompt...")
            config = calc_default_config | {
                'dataset_dir': "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/calculator",
                'augment_dir': "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/definite_horizon/augmented_standard/calculator",
                'prompt_batch_size': int(301 * size_factor),
                'filtering_batch_size': int(2501 * size_factor),
                # 'custom_dataset': custom_dataset,
                'm_arg_samples': 6,
                'num_of_processed_batches': 20,
                'k_positions': 15,
                'erase': False,
                # 'raw_arg_prompt': calculator_arg_prompt,
                'model_name': "LLAMA",
                'pos_threshold': 0.07,
                'generated_so_far': 886,
                'data_batch_size': 50
            }
            augment_calc(config)

        case "LLAMA Calculator trick 2":
            print("Augmenting calculator data with TRICK arg prompt...")
            config = calc_default_config | {
                'dataset_dir': "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/small_load/calculator",
                'augment_dir': "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/definite_horizon/augmented_prompttrick/calculator",
                'prompt_batch_size': int(101 * size_factor),
                'filtering_batch_size': int(512 * size_factor),
                # 'custom_dataset': custom_dataset,
                'm_arg_samples': 6,
                'num_of_processed_batches': 81+224,
                'k_positions': 20,
                'erase': False,
                'raw_arg_prompt': calculator_arg_prompt,
                'model_name': "LLAMA",
                'pos_threshold': 0.06,
                'generated_so_far': 4125,
                'num_of_processed_batches': 81,
                'data_batch_size': 50,
                'sampling_temperature': 0.9,
            }
            augment_calc(config)

        case "LLAMA Calculator trick":
            print("Augmenting calculator data with TRICK arg prompt...")
            config = calc_default_config | {
                'dataset_dir': "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/calculator",
                'augment_dir': "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/definite_horizon/augmented_prompttrick/calculator",
                'prompt_batch_size': int(101 * size_factor),
                'filtering_batch_size': int(512 * size_factor),
                # 'custom_dataset': custom_dataset,
                'm_arg_samples': 6,
                'num_of_processed_batches': 81+224,
                'k_positions': 20,
                'erase': False,
                'raw_arg_prompt': calculator_arg_prompt,
                'model_name': "LLAMA",
                'pos_threshold': 0.06,
                'generated_so_far': 4125,
                'num_of_processed_batches': 81,
                'data_batch_size': 50,
                'sampling_temperature': 0.6,
            }
            augment_calc(config)


        case "LLAMA WikiSearch trick2":
            print("Augmenting wikipedia data with SPECIAL arg prompt...", flush=True)
            config = wiki_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/wikiSearch',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/definite_horizon/augmented_prompttrick/wikiSearch2',
                'm_arg_samples': 3,
                'k_positions': 10,
                'prompt_batch_size': int(130 * size_factor),  # 135
                'filtering_batch_size': int(512 * size_factor),  # 2500
                'raw_arg_prompt': wikipedia_arg_prompt,
                'erase': False,
                # 'skip_files': ["0.csv"],
                'num_of_processed_batches': 0,
                'model_name': "LLAMA",    # GPTJ or LLAMA
                'pos_threshold': 0.06,
                'debug_level': 1,
                'sampling_temperature': 0.6,
                'generated_so_far': 0,
                # 'custom_dataset': data,
                'data_batch_size': 100
            }
            augment_wiki(config)

        case "LLAMA WikiSearch standard":
            print("Augmenting wikipedia data with STANDARD arg prompt...", flush=True)
            config = wiki_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/wikiSearch',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/definite_horizon/augmented_standard/wikiSearch',
                'm_arg_samples': 5,
                'k_positions': 10,
                'prompt_batch_size': int(290 * size_factor),  # 135
                'filtering_batch_size': int(512 * size_factor),  # 2500
                # 'raw_arg_prompt': wikipedia_arg_prompt,
                'erase': False,
                # 'skip_files': ["0.csv"],
                # 'num_of_processed_batches':10,
                'model_name': "LLAMA",    # GPTJ or LLAMA
                'pos_threshold': 0.06,
                'debug_level': 1,
                'sampling_temperature': 0.9,
                # 'generated_so_far': 3923,
                # 'custom_dataset': data,
                'data_batch_size': 100
            }
            augment_wiki(config)

        case "LLAMA Calendar":

            print(
                "Augmenting calendar data with STANDARD (duh...) arg prompt...", flush=True)
            config = calen_default_config | {
                'dataset_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/calendar',
                'augment_dir': '/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/definite_horizon/augmented_standard/calendar',
                'prompt_batch_size': int(680*size_factor),  # 135
                'filtering_batch_size': int(2500*size_factor),  # 2500
                # 'raw_arg_prompt': wikipedia_arg_prompt,
                'erase': True,
                # 'skip_files': ["0.csv"],
                # 'num_of_processed_batches':3,
                'model_name': "LLAMA",    # GPTJ or LLAMA
                'debug_level': 1,
                'pos_threshold': 0.07,
                # 'generated_so_far': 4093,
                # 'custom_dataset': data,
                'data_batch_size': 10000,
            }
            augment_calend(config)


        case "Calendar":
            print("Augmenting calendar data...")
            config = {
                'tool_name': "Calendar",
                'tool': Calendar,
                'tool_check_duplicates': identity_check,
                'raw_tool_prompt': prompt,
                'dataset_dir': "/vol/bitbucket/jg2619/data/preprocessed/big_load/calendar",
                'augment_dir': "/vol/bitbucket/jg2619/data/augmented2/calendar3",
                'max_args_length': 1,
                'max_data_length': 70,
                'm_arg_samples': 1,
                'k_positions': 8,
                'prompt_batch_size': 120,
                'data_batch_size': 1000,
                'filtering_batch_size': 2500,
                'extra_data_columns': ['date'],
                'new_data_columns': new_columns
            }

        case _:

            print("Expetiment name not recognized.")

    print("Congratulations, you have reached the end of the experiment.", flush=True)


if __name__ == "__main__":
    name = sys.argv[2] if len(sys.argv) > 2 else "Lightweight pipeline test"
    number_of_processes = device_count() or 1
    mp.spawn(experiment, args=(name, number_of_processes,), nprocs=number_of_processes)
    #experiment("Comparing max logits")
    #experiment("Lightweight pipeline test")
    #experiment("Continue GPTJ WikiSearch")
    #experiment("Unsorted. WARNING, DISCONTINUED")
    #experiment("Comparing probabilities")
    #experiment("Lightweight pipeline test")
    #experiment("Experiment 1")
    #experiment("LLAMA WikiSearch")
    #experiment("LLAMA Calculator")
    #experiment("LLAMA Calculator trick")
    #experiment("GPTJ Calendar")
    # init_wikisearch()
    # print(Calculator("2+3*4"))


