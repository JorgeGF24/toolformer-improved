from difflib import SequenceMatcher
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from transformers.deepspeed import HfDeepSpeedConfig
from torch.nn.utils.rnn import pad_sequence
import torch

import deepspeed

def left_pad_sequence(batch, padding_value):
    batch = [b.flip(dims=(-1,)) for b in batch]
    batch = pad_sequence(batch, padding_value=padding_value, batch_first=True)
    return batch.flip(dims=(-1,))

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

model_name = "LLAMA"

if model_name == "gpt-j":

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6B", cache_dir=cache_dir)

    tokenizer.add_tokens('[PAD]')
    tokenizer.pad_token = '[PAD]'

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, cache_dir=cache_dir
    ).cuda()
else:
    kwargs = {"cache_dir": cache_dir, 
              "token":"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",}

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                **kwargs)

    tokenizer.add_bos_token = False

    tokenizer.pad_token=tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-13b-hf", 
                                            padding_idx=tokenizer.pad_token_id,
                                            **kwargs)

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                config=config,
                                                **kwargs).cuda()

model.resize_token_embeddings(len(tokenizer))
model.eval()

ds_config = "/vol/bitbucket/jg2619/augmenting_llms/model_training/model_experiments/ds_conf3.json"
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

engine = deepspeed.initialize(model=model, config_params=dschf)


prompt = f"""Your task is to add calls to a Calendar API to a piece of text.
The API calls should help you get information required to complete the text.
You can call the API by writing "[Calendar()]".
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.****
Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.****
Input: [PAD]
Output:"""


prompt2 = [
"""Your task is to add calls to a Calendar API to a piece of text.
The API calls should help you get information required to complete the text.
You can call the API by writing "[Calendar()]".
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.****
Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.****
Input: The store is only open during the weekend, so today it is closed.
Output:""", 
"""Your task is to add calls to a Calendar API to a piece of text.
The API calls should help you get information required to complete the text.
You can call the API by writing "[Calendar()]".
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.****
Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.****
Input: The number of days from now until Christmas is 189.
Output:""", 
"""Your task is to add calls to a Calendar API to a piece of text.
The API calls should help you get information required to complete the text.
You can call the API by writing "[Calendar()]".
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.****
Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.****
Input: The current day of the week is Monday.
Output:"""]

# We want to test how efficient the model is when using past_key_values vs not using past_key_values

# First, we test the model without past_key_values

num_iter = 40
input_ids = torch.tensor(tokenizer.encode(prompt)).long().cuda()

# Make input ids tensor for prompt2
# First we encode each line in prompt2:
prompt2_encoded = [torch.tensor(tokenizer.encode(line)).long() for line in prompt2]
# Then we pad each line to the length of the longest line with pad_sequence:
prompt2_encoded = left_pad_sequence(prompt2_encoded, padding_value=tokenizer.pad_token_id).cuda()
attention_mask = prompt2_encoded != tokenizer.pad_token_id
# Zero out first values of each sequence:

# Decode last tokens of each sequence:
print("Last tokens of each sequence:", flush=True)
print("Decoded:")
print([tokenizer.decode(line[-1]) for line in prompt2_encoded])
print("Encoded:")
print([line[-1] for line in prompt2_encoded])

original_prompt2 = prompt2_encoded.clone()
cached_gen = []
no_cache_generation = []
no_cache_average = 0
cache_average = 0



generated = original_prompt2.clone()
attention_mask = generated != tokenizer.pad_token_id
past_key_values = None
# generated = torch.cat((prompt2_encoded, sample), dim=1)
for i in range(num_iter):
    start = time.time()
    input = model.prepare_inputs_for_generation(generated, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)
    output = model(**input)
    sample = output.logits[:, -1, :].argmax(-1).unsqueeze(1)
    generated = torch.cat((generated, sample), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones(sample.shape).cuda()), dim=1)
    past_key_values = output.past_key_values
    end = time.time()

    cache_average += (end - start) / num_iter
# Add decoded generation to cached generation:
cached_gen.extend([tokenizer.decode(line) for line in generated])

output = None
print(torch.cuda.memory_summary())
# Reset peak memory usage
torch.cuda.reset_peak_memory_stats()


prompt2_encoded = original_prompt2.clone()
attention_mask = prompt2_encoded != tokenizer.pad_token_id
for i in range(num_iter):
    start = time.time()

    input = model.prepare_inputs_for_generation(prompt2_encoded, attention_mask=attention_mask, use_cache=False)
    output = model(**input)
    prompt2_encoded = torch.cat((prompt2_encoded, output.logits[:, -1, :].argmax(-1).unsqueeze(1)), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1)).cuda()), dim=1)

    end = time.time()

    no_cache_average += (end - start) / num_iter
print(f"Average time taken without past_key_values: {no_cache_average}")
# Decode and print the generated text:
for sentence in prompt2_encoded:
    no_cache_generation.append(tokenizer.decode(sentence))



# Recurse tuple of past_key_values recursively and print shape of tensors when found:
def print_shape_of_tensors(tup):
    if isinstance(tup, tuple):
        for item in tup:
            print_shape_of_tensors(item)
    elif isinstance(tup, torch.Tensor):
        print(tup.shape)













print(torch.cuda.memory_summary())
print(f"Average time taken with past_key_values: {cache_average}")

# Check both generations are similar:
for g1, g2 in zip(cached_gen, no_cache_generation):
    print(f"Similarity: {SequenceMatcher(None, g1, g2).ratio()}")
    print(g1 == g2)
    print("")
