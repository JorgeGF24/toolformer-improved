import collections
import os
import random
import re

from datasets import load_dataset
import dateutil.parser as dparser
from dateutil.parser import ParserError
import time
import argparse
import nltk.data
import multiprocessing
from datetime import datetime
from transformers import AutoTokenizer
import traceback

import torch

from csv import DictWriter

from functools import wraps
import errno
import signal

import beartype
from beartype.typing import List

class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator

# Some statistics:

# Calculator:
# Calendar:
calend_data_outcome = {"skipped": 0, "date_present": 0}
calc_data_outcome = {"skipped": 0, "skipped_random":0, "choose_random": 0, "operation_combination":0, "operator_present": 0, "keywords": 0, "operator_and_keywords": 0}


def calend_available(
    url: str
) -> datetime:
    """
    Returns wether the calendar API could be used

    :param data: from load_dataset, assumes ['text'] is available
    :param tokenizer: Tokenizer to tokenize data
    :return: AvailableAPIs
    """
    url = url.replace("/", " ")
    try:
        date = dparser.parse(url, fuzzy=True, ignoretz=True)
        calend_data_outcome["date_present"] += 1
        # Check if date is after october 2019 or before the year 1200
        if date > datetime(2019, 10, 1) or date < datetime(1200, 1, 1):
            calend_data_outcome["skipped"] += 1
            return None, "skipped"

        return date.strftime("%Y-%m-%d"), "date_present"
    except (ValueError, OverflowError):
        calend_data_outcome["skipped"] += 1
        return None, "skipped"

def calc_available(
    data: List[str]
) -> List[str]:
    data_outcome = {"skipped": 0, "skipped_random":0, "choose_random": 0, "operation_combination":0, "operator_present": 0, "keywords": 0, "operator_and_keywords": 0}

    """
    :param data: list of sentences
    :return: List of sentences that contain a calculation
    """
    # In case we need a different version, found this here:
    # https://stackoverflow.com/questions/28198370/regex-for-validating-correct-input-for-calculator
    choose_3_sentences = False
    phrases = ["equals", "equal to", "total of", "average of"]
    phrases_regex = '|'.join(map(re.escape, phrases))
    # Phrases followed by a number, decimal or with commas
    equals_pattern = rf'({phrases_regex})\s*[$€¥£]?\d'

    available = []
    # deque that stores running counter of numbers in the last 3 sentences. 
    deque_len = 3 if choose_3_sentences else 1
    nums = collections.deque(deque_len*[0], deque_len)
    for i, sentence in enumerate(data):
        #print()
        #print("OPERATING AND RELATING")
        operators = bool(re.search(r"\d[$€¥£]?\s*=\s*[$€¥£]?\d", sentence))
        equals = bool(re.search(equals_pattern, sentence))

        
        if not (operators or equals):
            #print("NO OPERATORS OR EQUALS")
            words = sentence.split(" ")
            numbers = []
            for word in words:
                if word.replace(".", "", 1).isnumeric():
                    pass 
                elif word[:-1].replace(".", "", 1).isnumeric():
                    word = word[:-1]  # remove commas, full stops, units, etc.
                elif word[1:].replace(".", "", 1).isnumeric():
                    word = word[1:] # remove starting currency symbols
                else:
                    continue
                try:
                    num = float(word)
                except ValueError:
                    continue
                numbers.append(num)

            nums.append(len(numbers))
            is_oper_combi = False
            if sum(nums) >= 3:
                #print("THREE NUMBERS MIN")
                if sum(nums) < 10:
                    # Check if any of the numbers in words can be combined with +, -, *, / to produce a number in words
                    for num1 in numbers:
                        for num2 in numbers:
                            results = [num1 + num2, num1 - num2, num1 * num2]
                            if num2 != 0:
                                results.append(num1 / num2)
                            if any(result in numbers for result in results):
                                data_outcome['operation_combination'] += 1
                                is_oper_combi = True
                                break
                        if is_oper_combi:
                            break
                if not is_oper_combi and random.randint(0, 999) != 0:
                    data_outcome['skipped_random'] += 1
                    continue
                else:
                    data_outcome['choose_random'] += 1
            else:
                #print("NOT ENOUGH NUMBERS")
                data_outcome['skipped'] += 1
                continue
        else:
            if operators and equals:
                data_outcome['operator_and_keywords'] += 1
            elif equals:
                data_outcome['keywords'] += 1
            elif operators:
                data_outcome['operator_present'] += 1
            else:
                data_outcome['skipped'] += 1
                continue

        
        if choose_3_sentences:
            text = ""
            for j in range(max(0, i-2), i+1):
                text += data[j] + " "
        else:
            text = sentence + " "
        available.append(text[:-1])

    return available if len(available) > 0 else [], data_outcome

sentence_split = nltk.data.load('tokenizers/punkt/english.pickle', cache=False)

cache_dir = None
cache_option = {"cache_dir": cache_dir} if cache_dir else {} 

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", **cache_option)

space = tokenizer.encode(" ")

def break_sentence(sentence):
    # Tokenize sentence and break into chunks of 64 tokens
    tokens = tokenizer.encode(sentence, truncation=True, max_length=512, return_overflowing_tokens=True)

    tokens[-1] = tokens[-1] + space

    return tokens

@timeout(0.5)
def task(data, check_calc= True, check_calend= True, check_wiki= True):
    date = None
    outcome = "skipped"
    if check_calend:
        date, outcome = calend_available(data['url'])
        # get date from datetime object
    text = data['raw_content']
    # split text into sentences, which can be separated by full stops or newlines:
    text = sentence_split.tokenize(text)
    # make every sentence end with a period
    tokenized_sentences = []
    for sentence in text:
        tokenized_sentences += break_sentence(sentence)

    sentences = []
    current_sentence = []
    for sentence in tokenized_sentences:
        if len(current_sentence) == 0 and len(sentence) < 30:
            current_sentence += sentence
        elif len(current_sentence) > 0:
            current_sentence += sentence
            if len(current_sentence) > 100:
                sentences.append(tokenizer.decode(current_sentence))
                current_sentence = []
        else:
            sentences.append(tokenizer.decode(sentence))

    text = sentences

    calc_output = None
    if check_calc:
        calc_output, stats = calc_available(text)
    # Ouput file will have columns;
    # url, text, title, date_download, digest, length, nlines, source_domain, cc_segment, original_nlines, original_length, language, language_score, perplexity, bucket
    
    def data_row(text, date=None):
        row = {
            'url': data['url'],
            'text': text,
            'title': data['title'],
            'date_download': data['date_download'],
            'digest': data['digest'],
            'length': data['length'],
            'nlines': data['nlines'],
            'source_domain': data['source_domain'],
            'cc_segment': data['cc_segment'],
            'original_nlines': data['original_nlines'],
            'original_length': data['original_length'],
            'language': data['language'],
            'language_score': data['language_score'],
            'perplexity': data['perplexity'],
            'bucket': data['bucket'],
            'source_file': data['source_file']
        }
        if date:
            row['date'] = date
        return row
    
    calend_data = [data_row(sentence, date) for sentence in text] if date else []
    calc_data = [data_row(sentence) for sentence in calc_output] if calc_output else []
    wiki_data = [data_row(sentence) for sentence in text] if check_wiki else []

    return (calc_data, calend_data, wiki_data), stats, outcome

def worker(data, check_calc= True, check_calend= True, check_wiki= True):
    try:
        result, stats, outcome = task(data, check_calc, check_calend, check_wiki)
    except Exception as e:
        # stack trace:
        print("Exception in worker function")
        print(e)
        traceback.print_exc()
        return ([], [], []), {"skipped": 0, "skipped_random":0, "choose_random": 0, "operation_combination":0, "operator_present": 0, "keywords": 0, "operator_and_keywords": 0}, "skipped"

    return result, stats, outcome



erase = True


if __name__ == "__main__":
    print("HELLO")

    cache_dir = None
    dataset_dir = "/home/tromero_client/ccnet_filenames"
    processed_dir = "/home/tromero_client/ccnet_filenames/processed"  # "/vol/bitbucket/jg2619/data/preprocessed/big_load/"

    def file_name(tool, id):
        return f"{processed_dir}{tool}/{id}.csv"

    file_batch_size = 30   # number of files to load at a time
    file_batch = 0
    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.json')]

    dataset = load_dataset(dataset_dir, split="train", data_files = file_list[:file_batch_size], cache_dir = cache_dir)
    iter_data = iter(dataset)
    print("Loaded dataset")
    # Print rfirst row:

    tools = ["calculator", "calendar", "wikiSearch"]
    
    written_examples = {tool: 0 for tool in tools}
    new_data = {tool: list() for tool in tools}
    start_time = time.process_time()
    start_count = -1

    max_perplexity = 1000

    sample_size = 150000
    max_file_lines = 50000

    # Ouput file will have columns;
    # url, text, title, date_download, digest, length, nlines, source_domain, cc_segment, original_nlines, original_length, language, language_score, perplexity, bucket
    field_names = {"calculator":['url', 'text', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket', 'source_file']}
    field_names["calendar"] = field_names["calculator"][:2] + ['date'] + field_names["calculator"][2:]
    field_names["wikiSearch"] = field_names["calculator"]
    # And a shorter file with:
    # url, text, title

    def counter_not_done():
        for element in written_examples.values():
            if element < sample_size:
                return True
        return False
    
    print("Empieza movida")

    read_rows = 0
    file_ids = {tool:0 for tool in tools}
    current_file_rows = {tool:0 for tool in tools}
    written_rows = {tool:0 for tool in tools}
    # check if the files calendar_i.csv and calculator_i.csv exist for i in range(num_processes) and create them if not with field_names as the header

    for tool in tools:
        # Check if the processed_dir/tool exists and create it recursively if not
        if not os.path.exists(processed_dir + tool):
            os.makedirs(processed_dir + tool)
        if erase:
            # delete all files in the folder
            for file in os.listdir(processed_dir + tool):
                os.remove(f"{processed_dir}{tool}/{file}")
        if not os.path.exists(file_name(tool, file_ids[tool])):
            with open(file_name(tool, file_ids[tool]) , 'w') as f:
                writer = DictWriter(f, fieldnames=field_names[tool], escapechar='\\')
                writer.writeheader()

    data = next(iter_data, None)
    read_rows += 1
    while data['perplexity'] > max_perplexity:
        print(f"Skipping {data['url']} with perplexity {data['perplexity']}")
        read_rows += 1
        data = next(iter_data, None)

    processed_rows = 1
    while data != None or counter_not_done():
        # store output
        out = worker(data, written_examples["calculator"]<sample_size, written_examples["calendar"]<sample_size, written_examples["wikiSearch"]<sample_size)
        output_data, stats, calend_outcome = out
        
        calend_data_outcome[calend_outcome] += 1
        for key in stats.keys():
            calc_data_outcome[key] += stats[key]

        for i, tool in enumerate(tools):
            #print(f"Tool: {tool}")
            new_data[tool] += output_data[i]
            #print(len(new_data[tool]))

            if len(output_data[i]) > 0:
                pass#print(f"Found {len(output_data[i])} {tool} examples")

        if len(output_data[0]) + len(output_data[1]) + len(output_data[2]) == 0:
            pass#print("Found no examples")

        # Next data point
        data = next(iter_data, None)
        read_rows += 1
        while data is not None and data['perplexity'] > max_perplexity:
            print(f"Skipping {data['url']} with perplexity {data['perplexity']}")
            read_rows += 1
            data = next(iter_data, None)
        if data is None:
            file_batch += 1
            if file_batch*file_batch_size < len(file_list):
                print(f"Loading {file_batch}th batch of the dataset")
                dataset = load_dataset(dataset_dir, cache_dir=cache_dir, data_files = file_list[file_batch * file_batch_size:(file_batch + 1) * file_batch_size], split="train")
                #dataset.set_format("torch")
                data_iter = iter(dataset)
                data = next(data_iter, None)
                continue
            break
        
        # start new process
        args = (data, True, False, False) #written_examples[tool] < sample_size for tool in tools]
        #print("Type of args")
        #print(type(args))
        processed_rows += 1

        # write output to file
        for tool in tools:  
            size_new_examples = len(new_data[tool])
            if size_new_examples > 250:
                with open(file_name(tool, file_ids[tool]), 'a') as f:
                    # write all elements in list new_calculator_data to the csv file:
                    dictwriter_object = DictWriter(f, fieldnames=field_names[tool], escapechar='\\')

                    for row in new_data[tool]:
                        # Pass the dictionary as an argument to the Writerow()
                        dictwriter_object.writerow(row)

                    written_examples[tool] += size_new_examples
                    current_file_rows[tool] += size_new_examples
                    new_data[tool] = list()
                if current_file_rows[tool] > max_file_lines: #
                    file_ids[tool] += 1
                    current_file_rows[tool] = 0

                    if erase or not os.path.exists(file_name(tool, file_ids[tool])):
                        with open(file_name(tool, file_ids[tool]), 'w') as f:
                            writer = DictWriter(f, fieldnames=field_names[tool], escapechar='\\')
                            writer.writeheader()
            if read_rows%100 == 0:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("HIGH TESTING")
                print(f"Processed {read_rows} rows in {time.process_time() - start_time} seconds")
                for tool_name in tools:
                    print(f"Found {written_examples[tool_name]} {tool_name} examples")
                print(f"Skipped {calc_data_outcome['skipped']} calculator examples as they failed to meet the criteria")
                print(f"Skipped {calc_data_outcome['skipped_random']} calculator examples that had numbers")
                print(f"Chose {calc_data_outcome['choose_random']} random calculator examples that had numbers")
                print(f"Chose {calc_data_outcome['operator_present']} calculator examples that had operators")
                print(f"Chose {calc_data_outcome['keywords']} calculator examples that had keywords")
                print(f"Chose {calc_data_outcome['operator_and_keywords']} calculator examples that had both operators and keywords")
                print(f"Chose {calc_data_outcome['operation_combination']} calculator examples that had three numbers that could be operated into each other")
                print(f"Calendar skipped {calend_data_outcome['skipped']} examples")
                print(f"Calendar chose {calend_data_outcome['date_present']} examples")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print()
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("LLEGAMOS??????")

    print(f"Processed {processed_rows} rows in {time.process_time() - start_time} seconds")
    for tool in tools:
        print(f"Found {written_examples[tool]} {tool} examples")

    print(f"Skipped {calend_data_outcome['skipped']} calendar examples")
    print(f"Chose {calend_data_outcome['date_present']} calendar examples")
    print(f"Skipped {calc_data_outcome['skipped']} calculator examples as they failed to meet the criteria")
    print(f"Skipped {calc_data_outcome['skipped_random']} calculator examples that had numbers")
    print(f"Chose {calc_data_outcome['choose_random']} random calculator examples that had numbers")
    print(f"Chose {calc_data_outcome['operator_present']} calculator examples that had operators")
    print(f"Chose {calc_data_outcome['keywords']} calculator examples that had keywords")
    print(f"Chose {calc_data_outcome['operator_and_keywords']} calculator examples that had both operators and keywords")
    print(f"Chose {calc_data_outcome['operation_combination']} calculator examples that had three numbers that could be operated into each other")
    
    
    # Save stats to a stats file
    with open("stats.txt", "w") as f:
        f.write(f"Processed {processed_rows} rows in {time.process_time() - start_time} seconds\n")
        for tool in tools:
            f.write(f"Found {written_examples[tool]} {tool} examples\n")
        f.write(f"Skipped {calend_data_outcome['skipped']} calendar examples\n")
        f.write(f"Chose {calend_data_outcome['date_present']} calendar examples\n")
        f.write(f"Skipped {calc_data_outcome['skipped']} calculator examples as they failed to meet the criteria\n")
        f.write(f"Skipped {calc_data_outcome['skipped_random']} calculator examples that had numbers\n")
        f.write(f"Chose {calc_data_outcome['choose_random']} random calculator examples that had numbers\n")
        f.write(f"Chose {calc_data_outcome['operator_present']} calculator examples that had operators\n")
        f.write(f"Chose {calc_data_outcome['keywords']} calculator examples that had keywords\n")
        f.write(f"Chose {calc_data_outcome['operator_and_keywords']} calculator examples that had both operators and keywords\n")
        f.write(f"Chose {calc_data_outcome['operation_combination']} calculator examples that had three numbers that could be operated into each other\n")