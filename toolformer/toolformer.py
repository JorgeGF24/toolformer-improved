from imp import reload
import math
import os
import re

from collections import namedtuple
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor as Tensor 
 
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

from beartype import beartype
from beartype.typing import Callable, Optional, Union, List, Tuple, Dict

from tqdm import tqdm

import logging

pad_sequence = partial(pad_sequence, batch_first=True)

def left_pad_sequence(batch, padding_value):
    batch = [b.flip(dims=(-1,)) for b in batch]
    batch = pad_sequence(batch, padding_value=padding_value)
    return batch.flip(dims=(-1,))

API_START_TOKEN = ' ['
API_START_ID = None   # WILL BE UPDATED WHEN TOOLFORMER INITS
API_END_TOKEN = ']'
API_END_ID = None     # WILL BE UPDATED WHEN TOOLFORMER INITS
ARG_GEN_STOP_TOKENS = [']', ' ]', ')]', ' )]', '.]', '].', '],', ')', ')=', '→', '];',]  # Check for particular tokenizers which strings will be encoded in 1 single token.
ARG_GEN_STOPPERS = None    # WILL BE UPDATED WHEN TOOLFORMER INITS
API_ARGS_END = [')','.)',')]',' )',').']
PAD_TOKEN = '[PAD]'
PAD_ID = 50400        # SET TO PAD ID FROM TOKENIZER
DELIMITER_TOKEN = '→'
DELIMITER_ID = None
TOKENS_UNTIL_PROMPT_INPUT = None  # This counts how many tokens until the [PAD] token in the prompt to substitute the data for.
DECODE = None
ENCODE = None
DEVICE = 'cpu'
TRUNCATE_LENGTH = 100
MAX_RESPONSE_LENGTH = 50
MASKED_ARG_GENERATION = True
LLAMA = False
BOS_ID = 1
FILTER_THRESHOLD_EXPERIMENT = False
FILTER_STEP = 1/5  # the denominator is the number of tokens to consider before filtering to 0


CACHE_DIR = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"


# tensor helpers


def log(t, eps=1e-20): return t.clamp(min=eps).log()


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1, eps=1e-10):
    # Returns flat vector
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim=dim)


# the main contribution of the paper is simply the filtering equations presented in section 2


def default_weight_fn(t):
    return (1. - t * FILTER_STEP).clamp(min=0.)

def softer_weight_fn(t):
    # following the formula in section 4.1 - however, not sure what w_s is in the denominator
    # if t stands for each timestep, this would also mean within 5 tokens it would diminish to 0?
    return (1. - t * FILTER_STEP).clamp(min=0.)/2+0.5


def get_pred_prob(token_ids, logits):
    token_ids = rearrange(token_ids, 'b n -> b n 1')
    probs = logits.softmax(dim=-1)
    correct_token_id_pred_prob = probs.gather(-1, token_ids)
    return correct_token_id_pred_prob.squeeze(-1)


def get_arange_start_at_token_id(
    token_ids: torch.Tensor,
    token_id: int,
    null_mask_id: int = -1
):
    is_token_id_mask = token_ids == token_id
    arange = (is_token_id_mask.cumsum(dim=-1) > 0).cumsum(dim=-1)
    before_token_mask = arange == 0
    arange = arange - 1
    arange = arange.masked_fill(before_token_mask, null_mask_id)
    return arange

# Get arange start at position:
def get_arange_start_at_position(
    token_ids: torch.Tensor,  # b*s tensor with token ids
    positions: torch.Tensor,   # b*1 tensor with position for each text in batch
    null_mask_id: int = -1
):
    position_mask = torch.zeros(token_ids.shape)
    batch_indices = torch.arange(token_ids.shape[0]).unsqueeze(1)
    position_mask[batch_indices,positions.unsqueeze(1)] = 1
    arange = position_mask.cumsum(dim=-1).cumsum(dim=-1)
    before_token_mask = arange == 0
    arange = arange - 1
    arange = arange.masked_fill(before_token_mask, null_mask_id)
    arange = arange.masked_fill((token_ids == PAD_ID).cpu(), null_mask_id)
    # Count non zero elements per row:
    total_weights = 1/(arange != null_mask_id).sum(dim=1)
    return arange, total_weights


def weight_and_mask(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    weighting_fn: Callable = default_weight_fn,
    null_mask_id: int = -1,
    device: str = 'cpu'
):
    t, denoms = get_arange_start_at_position(token_ids, positions, null_mask_id)
    weights = weighting_fn(t)
    return weights.masked_fill(t == null_mask_id, 0.).to(device) # * denoms.unsqueeze(1)


# datasets and dataloaders

# for bootstrapping the initial datasets with api calls
# as well as for the final finetuning
 

@beartype
class PromptDataset(Dataset):
    def __init__(
        self,
        data: Union[List[str],List[List[int]]],
        tokenized_raw_prompt: List[int],
        tokenized_arg_prompt: List[int],
        substitute_index: int,
        tokenizer_encode: Callable,
        data_tokenized: bool = False,
        new_to_old_idx: List[int] = None,
    ):
        self.tokenized_raw_prompt = tokenized_raw_prompt
        if MASKED_ARG_GENERATION:
            self.tokenized_arg_prompt = tokenized_arg_prompt
        self.substitute_index = substitute_index
        
        self.encode = tokenizer_encode

        if not data_tokenized:
            self.data = [self.encode(datum, truncation=True, max_length=TRUNCATE_LENGTH) for datum in data]
        
        # Sort data by increasing length
        idx_order, self.data = zip(*sorted(enumerate(self.data), key=lambda x: len(x[1])))
        new_to_old_idx.sort(key=lambda x: idx_order.index(x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tokens = self.data[idx]
        prompt_tokens = self.tokenized_raw_prompt[:]
        prompt_tokens[self.substitute_index:self.substitute_index+1] = data_tokens

        #print("getting item from prompt...")
        # print(token_ids)
        # print(len(token_ids))
        # print(DECODE(token_ids))
        #print(f"Last token: {DECODE(token_ids[-1])}")
        
        # arg_prompt_data = Tensor(self.tokenized_arg_prompt+data_tokens).long() if MASKED_ARG_GENERATION else None
        return Tensor(prompt_tokens).long(), Tensor(data_tokens).long(), len(data_tokens), idx


def prompt_collate_fn(data):
    prompts, data, data_lengths, indices = zip(*data)

    return prompts, data, Tensor(data_lengths), Tensor(indices, device=DEVICE)


def PromptDataloader(ds: Dataset, *args, **kwargs):
    collate_fn = partial(prompt_collate_fn)
    return DataLoader(ds, *args, collate_fn=collate_fn, **kwargs)

# TOOL NAME DECODE
@beartype
class APICallDataset(Dataset):
    def __init__(
        self,
        data: List[torch.Tensor],
        call_response_pos_idxs: List[Tuple],
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        device: torch.device,
        tool_name: str,
    ):
        self.data = data
        self.call_response_pos_idxs = call_response_pos_idxs
        self.device = device

        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.cache_dir = CACHE_DIR

        self.tool_name = tool_name

    def __len__(self):
        return len(self.call_response_pos_idxs)

    def __getitem__(self, idx):
        tokenized_args, tokenized_response, pos, data_idx, call_idx, _ = self.call_response_pos_idxs[idx]

        start_str = API_START_TOKEN + self.tool_name + "("
        arg_str = self.decode(tokenized_args)

        bare_end_str = f"){API_END_TOKEN}"
        no_resp_end_str = {
            'filtering_format': bare_end_str + ".\n\n",
            'output_clean_format': bare_end_str
        }
        bare_response = f"){DELIMITER_TOKEN} {self.decode(tokenized_response)}{API_END_TOKEN}"
        response_str = {
            'filtering_format': bare_response + ".\n\n",
            'output_clean_format': bare_response,
        } 

        call = {key: self.encode(start_str + arg_str + end_str, return_tensors='pt').squeeze(0).to(self.device).long() for key, end_str in no_resp_end_str.items()}
        call_resp = {key: self.encode(start_str + arg_str + end_str, return_tensors='pt').squeeze(0).to(self.device).long() for key, end_str in response_str.items()}
        data_with_call = torch.cat([call['filtering_format'], self.data[data_idx].to(self.device)])
        data_with_call_resp = torch.cat([call_resp['filtering_format'], self.data[data_idx].to(self.device)])

        # Positions of where the calls w and w/o responses end
        len_call = call['filtering_format'].shape[0]
        pos_call_end = pos + len_call
        len_resp = call_resp['filtering_format'].shape[0]
        pos_resp_end = pos + len_resp
        
        return self.data[data_idx], data_with_call, data_with_call_resp, pos, pos_call_end, pos_resp_end, call['output_clean_format'].cpu(), call_resp['output_clean_format'].cpu(), data_idx, call_idx

def API_collate_fn(data):
    global PAD_ID, LLAMA, BOS_ID
    data_plain, data_api, data_api_resp, pos, pos_call_end, pos_resp_end, calls, call_resps, data_indices, call_indices = zip(*data)

    position_triplet = tuple([Tensor(x) for x in (pos, pos_call_end, pos_resp_end)])
    if LLAMA:
        # Add 1 to the positions to account for bos that we'll add during the padding stage
        position_triplet = tuple([x+1 for x in position_triplet])

    return (data_plain, data_api, data_api_resp), position_triplet, tuple([list(calls), list(call_resps)]), data_indices, call_indices

def APIDataloader(ds: Dataset, *args, **kwargs):
    collate_fn = partial(API_collate_fn)
    return DataLoader(ds, *args, collate_fn=collate_fn, **kwargs)


class FinetuneDataset(Dataset):
    def __init__(
        self,
        tokens: torch.Tensor
    ):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def FinetuneDataloader(ds: Dataset, *args, **kwargs):
    return DataLoader(ds, *args, collate_fn=partial(pad_sequence, padding_value=PAD_ID), **kwargs)

# classes
GeneratedArgsCall = namedtuple('GeneratedArgsResults', [
    'tokenized_args',
    'position',
    'data_idx',
    'call_idx'
])

@beartype
class ArgString():
    def __init__(self, arg: torch.Tensor):
        self.arg = arg

    def __str__(self):
        return self.arg
    
    # modify value of arg
    def update(self, arg: torch.Tensor):
        self.arg = arg

    # get
    def get(self) -> torch.Tensor:
        return self.arg

@beartype
class Toolformer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        tool_name: str,
        tool: Callable,
        tool_check_duplicates: Callable,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        raw_tool_prompt: str,
        raw_arg_prompt: str=None,
        preprocess_args: Callable=None,
        pad_token=PAD_TOKEN,
        api_start_token=API_START_TOKEN,
        api_end_token=API_END_TOKEN,
        filter_threshold=1.,
        pos_threshold: float = 0.05,
        prompt_batch_size:int=20,
        filtering_batch_size=25,
        max_arg_length=20,
        max_response_length=50,
        k_positions=5,
        m_arg_samples=5,
        sampling_temperature=1.,
        max_data_length=100,
        model_seq_len=2048,  # POTENTIALLY REMOVE
        tokenizer_batch_decode: Callable = None,
        softer_weight:bool=False,
        debug_level:int=1,  # 0 is only info, 1 is more detail, 2 is excruciating detail
        log_dir:str="/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer-luci/lost_logs",
        using_llama:bool=False,
        bos_id:int=1,
        experiment_config:Dict={},
        arg_gen_stoppers:torch.Tensor=None,
        batch_size_decrease_step:int=7,
        max_cuda_error_count:int=50,
        **kwargs
    ):
        super().__init__()

        for key in kwargs.keys():
            print(f"WARNING: {key} is not used in Toolformer")

        global FILTER_STEP, BOS_ID, LLAMA, MASKED_ARG_GENERATION, API_START_TOKEN, API_START_ID, API_END_TOKEN, API_END_ID, ARG_GEN_STOPPERS, DELIMITER_ID, PAD_ID, PAD_TOKEN, DECODE, ENCODE, DEVICE, TRUNCATE_LENGTH, MAX_RESPONSE_LENGTH

        self.model = model
        self.model_seq_len = model_seq_len
        self.device = next(self.model.parameters()).device
        DEVICE = self.device
        
        LLAMA = using_llama
        if using_llama:
            self.bos_id = bos_id
            BOS_ID = bos_id
            FILTER_STEP = 1/7

            
        self.debug_level = debug_level
        # Create log dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # count files in log dir
        i = len(os.listdir(log_dir))
        logging.basicConfig(filename=f'{log_dir}/{i}.log', level=logging.DEBUG if debug_level>0 else logging.INFO, format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S  ')
        print(f"Logging to {log_dir}/{i}.log")
        logging.info("Experiment config:")
        for key, value in experiment_config.items():
            logging.info(f"{key}: {value}")

        self.prompt_batch_size = prompt_batch_size
        self.filtering_batch_size = filtering_batch_size
        self.BATCH_SIZE_DECREASE_STEP = batch_size_decrease_step
        self.MAX_CUDA_ERROR_COUNT = max_cuda_error_count
        
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        DECODE = tokenizer_decode # FOR DEBBUGING
        ENCODE = tokenizer_encode # FOR DEBBUGING
        TRUNCATE_LENGTH = max_data_length
        MAX_RESPONSE_LENGTH = max_response_length

        if tokenizer_batch_decode is None:
            tokenizer_batch_decode = lambda data_list: [self.decode(data) for data in data_list]
        self.batch_decode = tokenizer_batch_decode
        self.encode_to_tensor = lambda s: Tensor(
            tokenizer_encode(s)).long()

        self.filter_threshold = filter_threshold
        self.pos_threshold = pos_threshold
        print("pos_threshold:", self.pos_threshold)
        self.max_arg_length = max_arg_length
        self.m_arg_samples = m_arg_samples
        self.k_positions = k_positions
        self.sampling_temperature = sampling_temperature
        
        #self.API_START_TOKEN = API_START_TOKEN
        #self.API_END_TOKEN = API_END_TOKEN
        #self.api_response_delimiter = api_response_delimiter
        
        PAD_TOKEN = pad_token
        PAD_ID = ENCODE(pad_token)
        assert len(PAD_ID) == 1
        PAD_ID = PAD_ID[0]

        if LLAMA and api_start_token[0] == ' ':
            # Remove space from api start token
            api_start_token = api_start_token[1:]
        API_START_TOKEN = api_start_token
        API_START_ID = ENCODE(api_start_token)
        assert len(API_START_ID) == 1
        API_START_ID = API_START_ID[0]

        API_END_TOKEN = api_end_token
        API_END_ID = ENCODE(api_end_token)
        assert len(API_END_ID) == 1
        API_END_ID = API_END_ID[0]

        if arg_gen_stoppers is not None:
            ARG_GEN_STOPPERS = arg_gen_stoppers.long().to(self.device)
        else:
            ARG_GEN_STOPPERS = Tensor([ENCODE(token)[-1] for token in ARG_GEN_STOP_TOKENS if len(ENCODE(token))==1], device=self.device).long()
        DELIMITER_ID = tokenizer_encode(DELIMITER_TOKEN)
        assert len(DELIMITER_ID) == 1
        DELIMITER_ID = DELIMITER_ID[0]

        self.tool_name = tool_name
        self.tool = tool
        self.tokenized_tool_name = tokenizer_encode(tool_name)
        self.tool_check_duplicates = tool_check_duplicates
        self.preprocess_args = preprocess_args

        self.raw_tool_prompt = raw_tool_prompt
        
        if not raw_arg_prompt:
            MASKED_ARG_GENERATION = False
            raw_arg_prompt = raw_tool_prompt
        else:
            logging.info("MASKED ARG GENERATION")
            MASKED_ARG_GENERATION = True
        raw_tool_prompt = raw_tool_prompt.replace("[PAD]", pad_token)
        self.raw_arg_prompt = raw_arg_prompt
        self.tokenized_raw_prompt = tokenizer_encode(raw_tool_prompt)
        self.tokenized_raw_arg_prompt = tokenizer_encode(raw_arg_prompt)
        if using_llama:
            self.tokenized_raw_prompt = [self.bos_id] + self.tokenized_raw_prompt
            self.tokenized_raw_arg_prompt = [self.bos_id] + self.tokenized_raw_arg_prompt
        self.arg_prompt_len = len(self.tokenized_raw_arg_prompt)
        try:
            self.substitute_index = self.tokenized_raw_prompt.index(PAD_ID)
        except ValueError:
            print(self.raw_tool_prompt)
            print(f'there must be exactly one pad token `{PAD_TOKEN}` in your prompt in which to substitute the data to be annotated')

        self.weight_func = default_weight_fn if not softer_weight else softer_weight_fn
        self.tighter_loss = softer_weight


        print("INITed Toolformer")

    @torch.no_grad()
    def sample_API_positions(
        self,
        *,
        prompt_data_pad: torch.Tensor,   # right and left padded, centred at prompt_size X, pad-prompt-data-X-(output: data)
        prompt_size: int,   
        data_lengths: torch.Tensor,      # lengths of data in prompt_data_pad
        model: nn.Module = None,
        sampling_threshold=0.05,
    ):
        """
            Returns the top k positions in the data where the model is most likely to insert an API call. Index 0 means inserting it as the first token in the data.
        """
        if model is None:
            model = self.model

        batch_size, input_length = prompt_data_pad.shape

        # Logits from last prompt token (as model predicts next token)
        inputs = model.prepare_inputs_for_generation(prompt_data_pad[:,:-2], use_cache=False)
        logits = model(**inputs).logits[:, prompt_size-1:]

        logits = logits.softmax(dim=-1)[..., API_START_ID]
        pad_column = max(0, self.k_positions + 2 - torch.min(data_lengths, dim=0)[0].item())
        if (pad_column > 0):
            # Add column of 0s to logits
            logits = torch.cat([logits, torch.zeros((batch_size, pad_column), device = self.device)], dim = 1)

        mask = torch.arange(input_length-prompt_size - 1 + pad_column).unsqueeze(0) >= data_lengths.unsqueeze(1)-1
        logits = logits.masked_fill(mask.to(self.device), 0.)

        values, positions = logits.topk(dim=-1, k=self.k_positions, sorted = False)

        if FILTER_THRESHOLD_EXPERIMENT:
            return positions, values

        above_threshold_indices = (values >= sampling_threshold).int()
        if self.debug_level > 1:
            logging.debug(f"Logits: {logits}")
            logging.debug(f"Values: {values}")
            logging.debug(f"sampling_threshold: {sampling_threshold}")
            logging.debug(f"above_threshold_indices: {above_threshold_indices}")
        positions *= above_threshold_indices # Zero out bad values
        
        # COULD OMIT THIS SORT IN FUTURE
        # positions, _ = positions.sort(dim=-1, descending=True)

        max_call_count = above_threshold_indices.sum(dim=1).max()

        return positions, max_call_count
    
    @torch.no_grad()
    def generate_API_call_arguments(
        self,
        *,
        prompt_data: torch.Tensor,  # pad + prompts (with data substituted in [PAD]) + # + data + pad SO # are all aligned accross rows
        data_indices: torch.Tensor, # Index of data in original data list
        positions: torch.Tensor,    # Positions of API calls in data
        call_counter: int,
        temperature=0.9,
        random_finish_p: Union[float, int] = None,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[Dict, int, Dict]:
        
        global MASKED_ARG_GENERATION, API_START_ID, ARG_GEN_STOPPERS, PAD_ID, DECODE, ENCODE, LLAMA

        if attention_mask is None:
            attention_mask = prompt_data != PAD_ID
        attention_mask.to(self.device)

        past_key_values = None

        model_displacement = 1    # Should be 0 for models where forward at position i is logits for token at i. Llama and GPTJ return logits for i+1.
        batch_size, initial_prime_length = prompt_data.shape

        # Adjust per model: check positions of this variable. Some require second one to be -1.
        part_timers = {
            "prepare":0,
            "forward":0,
            "gumbel sample":0,
            "postprocess and stopping":0,
        }
        part_timers["prepare"] = -time.time()

        generated_calls = {}
        generated_counter = 0

        # lengthen the prime to the entire sequence length. 
        # We give room for adding arg_length arg tokens, tokens for tool name, space before call, start '[' and '(', end ')' and ']'
        # output = F.pad(prompt_data, (0, self.max_arg_length + len(self.tokenized_tool_name) + 5), value=PAD_ID)

        if False:
            # Add space before api call. We move the position cursor back 1 to account for this
            position_indices -= 1
            unspaced_tokens = output[batch_indices, position_indices].squeeze(dim=1)

            logging.debug(f"Unspaced tokens: {DECODE(unspaced_tokens)}")
            # Unbatched processing as dealing with strings
            for idx, token in enumerate(unspaced_tokens):
                word = DECODE(token.item())
                if word[-1] != ' ':
                    word += ' '
                tokens = ENCODE(word)
                for j, token in enumerate(tokens):
                    pos = position_indices[idx].item()
                    output[idx, pos+j] = token
                    # Increase position cursor for each added token
                    position_indices[idx] += 1

        tool_call_start = Tensor(self.encode(API_START_TOKEN + self.tool_name + '(')).long().to(self.device)
        tool_call_start = tool_call_start.unsqueeze(0).repeat(batch_size, 1)
        initial_prime_length += tool_call_start.shape[1]
        input_ids = torch.cat([prompt_data, tool_call_start], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(tool_call_start.shape, device = self.device)], dim=1)
        finished = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        loop_to_batch_idx = torch.arange(batch_size, device=self.device)

        # TODO move log to outside function where data lengths are available
        """
        if self.debug_level > 1:
            for idx in batch_indices:
                pos = position_indices[idx].item()
                logging.debug(f"{idx.item()}th Prime: {DECODE(output[idx,prompt_size:pos].squeeze(dim=0))}")
                logging.debug(f"{idx.item()}th Prime: {DECODE(output[idx].squeeze(dim=0))}")
                logging.debug(f"Samplig at position {DECODE(output[idx,pos-model_displacement].squeeze(dim=0))}, right after {DECODE(output[idx,pos-model_displacement-1].squeeze(dim=0))}")
        """

        finish_section = time.time()
        part_timers["prepare"] += finish_section
        part_timers["forward"] -= finish_section

        for sentence in input_ids:
            print(DECODE(sentence))

        # Add 2 to account for extra ) and ] tokens
        for _ in range(self.max_arg_length + 2):

            input = self.model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
            output = self.model(**input)
            last_logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values
            output = None

            finish_section = time.time()
            part_timers["forward"] += finish_section
            part_timers["gumbel sample"] -= finish_section

            # greedy sample (but could be made non-greedy)
            sampled = torch.ones(batch_size, device=self.device, dtype=torch.long) * PAD_ID
            sampled[~finished] = gumbel_sample(last_logits[~finished], temperature=temperature)

            finish_section = time.time()
            part_timers["gumbel sample"] += finish_section
            part_timers["postprocess and stopping"] -= finish_section

            # Concat sampled to input
            input_ids = torch.cat([input_ids, sampled.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, (~finished).reshape(attention_mask.shape[0],1).long()], dim=1)

            # If any sampled are </API> thats a call thats done
            # Random boolean tensor of same shape as sampled where each row is True with a probability of p:
            
            finished_samples = (torch.isin(sampled[~finished], ARG_GEN_STOPPERS))
            
            if random_finish_p:                                                                              
                # Finish some samples randomly, or with 100% prob for p = 1 in case argumnents should have 0 length
                random_finish = torch.rand(sampled.shape[0], device=self.device) < random_finish_p
                finished_samples = finished_samples | random_finish

            if finished_samples.any():
                for finished_idx in finished_samples.nonzero().squeeze(1):
                    # Args, api position, data index
                    data_index = data_indices[finished_idx].item()
                    batch_idx = loop_to_batch_idx[finished_idx].item()
                    call = GeneratedArgsCall(ArgString(input_ids[batch_idx,initial_prime_length:].cpu()), positions[finished_idx], data_index, call_counter)
                    call_counter += 1
                    generated_calls[str(data_index)] = generated_calls.get(str(data_index),[]) + [call]
                    generated_counter += 1
                    batch_indices = batch_indices[:-1]
                    finished[batch_idx] = True

                # Only continue generating for incomplete sequences 
                #input_ids = input_ids[~finished_samples]
                data_indices = data_indices[~finished_samples]
                positions = positions[~finished_samples]
                loop_to_batch_idx = loop_to_batch_idx[~finished_samples]

                #attention_mask = attention_mask[~finished_samples]
                # Reduce past_key values to only those that are still being generated
                # Past key value is a tuple of tuples of tensors. We descend down to each and remove the finished samples
                #past_key_values = tuple(tuple(key_value[~finished_samples] for key_value in past_key_value) for past_key_value in past_key_values)

            finish_section = time.time()
            part_timers["postprocess and stopping"] += finish_section
            part_timers["forward"] -= finish_section

            if generated_counter == batch_size:
                break
        
        part_timers["forward"] += finish_section

        # remove the last token in output (use as noop placeholder)
        # MMMM^^^????
        logging.debug(f"{data_indices.shape[0]}/{prompt_data.shape[0]} examples failed to finish sampling")
        logging.debug(f"These were:")
        for sentence in input_ids:
            logging.debug(f"{DECODE(sentence)}")
            arg = sentence[initial_prime_length:]
            logging.debug(f"{DECODE(arg)}")


        # Return list of tuples (tokenized API call, data index)
        return generated_calls, call_counter, part_timers
    
    
    @beartype
    @torch.no_grad()
    def filter_tokens_with_api_response(
        self,
        # token ids (batch, seq) of the original passage, without api calls
        tokens: torch.Tensor,
        # token ids (batch, seq) of the passage, but with the api call (but without a response filled in) - <api>tool1(x, y)</api>
        tokens_without_api_response: torch.Tensor,
        # token ids (batch, seq) of the passage with api call and the response - <api>tool1(x, y) → {response}</api>
        tokens_with_api_response: torch.Tensor,
        # the positions at which to start measuring loss for each sequence
        filter_positions: list[torch.Tensor],
        # the data indices of each sequence
        data_indices: torch.Tensor,
        # the call indices of each sequence
        call_indices: torch.Tensor,
        # the threshold at which to accept the sampled api call (tokens_with_api_response) for fine-tuning
        filter_threshold: float = 1.5,
        weighting_fn: Callable = default_weight_fn     # weighting function
    ) -> Dict:

        # validations

        assert all([*map(lambda t: t.dtype == torch.long, (tokens,
                tokens_with_api_response, tokens_without_api_response))]), f"tokens must be of type long, but are {tokens.dtype}, {tokens_with_api_response.dtype}, {tokens_without_api_response.dtype}"

        # auto set devices
        tokens, tokens_without_api_response, tokens_with_api_response = map(lambda t: t.to(
            self.device), (tokens, tokens_without_api_response, tokens_with_api_response))

        # Minus 1 here as we dont want predictions AFTER end of data
        output, output_without_api_response, output_with_api_response = map(
            partial(self.model, use_cache=False), 
            (tokens[:,:-1], tokens_without_api_response[:,:-1], tokens_with_api_response[:,:-1]))

        # Predictions are AFTER seeing token, so no prediction of first token
        tokens = tokens[:,1:]
        tokens_without_api_response = tokens_without_api_response[:,1:]
        tokens_with_api_response = tokens_with_api_response[:,1:]

        # derive all predicted prob of the actual next token id in sequence
        probs = get_pred_prob(tokens, output.logits)
        probs_without_api_response = get_pred_prob(
            tokens_without_api_response, output_without_api_response.logits)
        probs_with_api_response = get_pred_prob(
            tokens_with_api_response, output_with_api_response.logits)
        
        output = None
        output_without_api_response = None
        output_with_api_response = None

        weight_and_mask_fn = partial(weight_and_mask, weighting_fn=weighting_fn, device=self.device)

        if False:
            print(f"tokens {tokens[0]}")
            print(f"decoded tokens {DECODE(tokens[0])}")
            print(f"api positions {filter_positions[0]}")
            print(f"tokens without api response {tokens_without_api_response[0]}")
            print(f"decoded tokens without api response {DECODE(tokens_without_api_response[0])}")

        # derive the weighting - subtract 1 from the filter positions since the first token is not predicted
        weight = weight_and_mask_fn(tokens, positions=filter_positions[0]-1)
        weight_without_api_response = weight_and_mask_fn(
            tokens_without_api_response, positions=filter_positions[1]-1)
        weight_with_api_response = weight_and_mask_fn(
            tokens_with_api_response, positions=filter_positions[2]-1)
                
        # get the loss L for all three types of sequences

        def loss_fn(W, P):
            return (W * -log(P)).sum(dim=-1)
        
        #print(f"Output logits have argmax {output.logits[0].argmax(dim=-1).tolist()}")
        #print(f"Token ids are {tokens[0].tolist()}")
        #print(f"Output without api response logits have argmax {output_without_api_response.logits[0].argmax(dim=-1).tolist()}")
        #print(f"Token ids without api response are {tokens_without_api_response[0].tolist()}")
        #print(f"Output with api response logits have argmax {output_with_api_response.logits[0].argmax(dim=-1).tolist()}")
        #print(f"Token ids with api response are {tokens_with_api_response[0].tolist()}")

        #print(DECODE(tokens[0].tolist()))
        #print(DECODE(output.logits[0].argmax(dim=-1).squeeze().tolist()))

        loss = loss_fn(weight, probs)
        loss_without_api_response = loss_fn(
            weight_without_api_response, probs_without_api_response)
        loss_with_api_response = loss_fn(
            weight_with_api_response, probs_with_api_response)

        # calculate the main formula in the paper

        # loss+ = loss with api response
        # loss- = min(loss without api response, loss without api at all)

        if self.debug_level > 1: logging.debug(f"Losses are: {loss} // {loss_without_api_response} // {loss_with_api_response}")

        loss_plus = loss_with_api_response
        loss_minus = torch.minimum(loss_without_api_response, loss)

        loss_change = loss_minus - loss_plus

        if self.debug_level > 1: logging.debug(f"Loss change is: {loss_change}")

        
        if self.debug_level > 1:
            seq_len_1 = tokens.shape[1]
            seq_len_2 = tokens_without_api_response.shape[1]
            for i in range(tokens.shape[0]):
                data_diff = filter_positions[2][i] - filter_positions[0][i]
                API_diff = filter_positions[2][i] - filter_positions[1][i]
                logging.debug(f"Call id: {call_indices[i].item()} // Data id: {data_indices[i].item()} // position: {filter_positions[0][i].item()}")
                logging.debug(f"Filtering data:{DECODE(tokens[i].squeeze(0))}")
                logging.debug(f"Filtering data without resp:{DECODE(tokens_without_api_response[i].squeeze(0))}")
                logging.debug(f"Filtering data with resp:{DECODE(tokens_with_api_response[i].squeeze(0))}")
                # We print 3 columns, each showing the weights and tokens of each type:
                logging.debug("Probs - Weights - Data tokens // Probs - Weights - Only call tokens // Probs - Weights - Call and Response tokens")
                for j3 in range(tokens_with_api_response.shape[1]):
                    j1 = j3 - data_diff
                    j2 = j3 - API_diff
                    # The columns are equally spaced, padded with spaces
                    p1 = probs[i,j1].item() if j1 >= 0 and j1<seq_len_1 else 0.0
                    p2 = probs_without_api_response[i,j2].item() if j2 >= 0 and j2<seq_len_2 else 0.0
                    p3 = probs_with_api_response[i,j3].item()
                    w1 = int(10*weight[i,j1].item()) if j1 >= 0 and j1<seq_len_1 else 0.0
                    if w1 == 0:
                        w1 = "- "
                    w2 = int(10*weight_without_api_response[i,j2].item()) if j2 >= 0 and j2<seq_len_2 else "- "
                    if w2 == 0:
                        w2 = "- "
                    w3 = int(10*weight_with_api_response[i,j3].item())
                    if w3 == 0:
                        w3 = "- "
                    t1 = DECODE(tokens[i,j1].item()) if j1 >= 0 and j1<seq_len_1 else "-"
                    t2 = DECODE(tokens_without_api_response[i,j2].item()) if j2 >= 0 and j2<seq_len_2 else "-"
                    t3 = DECODE(tokens_with_api_response[i,j3].item())
                    logging.debug(f"{p1:.4e}  {w1:<2}:  '{t1:<10}' // {p2:.4e}  {w2:<2}:  '{t2:<10}' // {p3:.4e}  {w3:<2}:  '{t3}'")

                logging.debug("========== Losses ==========")
                logging.debug(f"Loss: {loss[i].item():.2f} // Loss without api response: {loss_without_api_response[i].item():.2f} // Loss with api response: {loss_with_api_response[i].item():.2f}")
                logging.debug(f"Loss change: {loss_change[i].item():.2f}")

        selected_mask = (loss_change >= filter_threshold).to('cpu')

        # now we can select and return the entries that survived the filtering stage
        # also returning the selected indices of the batch being processed
        # for finetuning the model into toolformer

        selected_indices = data_indices[selected_mask]
        call_indices = call_indices[selected_mask]

        ret = {
            'num_passed': selected_mask.sum().item(),
            'num_failed': (~selected_mask).sum().item(),
            'selected_indices': selected_indices,
            'call_indices': call_indices,
            'call_positions': filter_positions[0][selected_mask],
            'selected_mask': selected_mask,
            'loss_change': loss_change,
            'loss_selected': loss_change[selected_mask],
        }

        return ret

    def finetune(
        self,
        filtered_results: torch.Tensor
    ):
        self.model.train()

        if False:#isinstance(filtered_results, FilteredResults):
            filtered_results = filtered_results.filtered_tokens_without_api_response

        dataset = FinetuneDataset(tokens=filtered_results)
        dl = FinetuneDataloader(
            dataset, batch_size=self.finetune_batch_size, shuffle=True)

        for epoch in tqdm(range(self.finetune_epochs), desc='finetune epochs'):
            for batch in dl:
                inp, labels = batch[:, :-1], batch[:, 1:]

                logits = self.model(inp)
                logits = rearrange(logits, 'b n c -> b c n')

                loss = F.cross_entropy(
                    logits, labels, ignore_index=PAD_ID)
                loss.backward()

                print(f'loss: {loss.item()}')
                self.optimizer.step()
                self.optimizer.zero_grad()

        print(f'finished finetuning on {len(dataset)} filtered samples')

    @torch.no_grad()
    def compare_pos_sampling(
        self,
        data: List[str],
        model,
        encode,
    ):
        global FILTER_THRESHOLD_EXPERIMENT, PAD_ID, PAD_TOKEN
        FILTER_THRESHOLD_EXPERIMENT = True
        
        PAD_ID = encode(PAD_TOKEN)
        assert len(PAD_ID) == 1
        PAD_ID = PAD_ID[0]

        self.encode = encode
        self.tokenized_raw_prompt = encode(self.raw_tool_prompt)
        self.tokenized_raw_arg_prompt = encode(self.raw_arg_prompt)
        try:
            self.substitute_index = self.tokenized_raw_prompt.index(PAD_ID)
        except ValueError:
            print(f'there must be exactly one pad token `{PAD_TOKEN}` in your prompt in which to substitute the data to be annotated')


        dataset = PromptDataset(
            data=data,
            tokenized_raw_prompt=self.tokenized_raw_prompt,
            tokenized_arg_prompt=self.tokenized_raw_arg_prompt,
            substitute_index=self.substitute_index,
            tokenizer_encode=self.encode,
        )
        dl = PromptDataloader(
            dataset,
            batch_size=self.prompt_batch_size
        )

        values = []
        lengths = []
        
        # TODO: Dl has changed outputs. Should be padded and processed before being passed to sample_API_positions
        for prompt_tuple, data_pad, data_list, data_length, data_indices in dl:

            prompts, arg_prompts = prompt_tuple
            prompt_size = prompts.shape[1]
            prompt_data_pad = torch.cat((prompts, data_pad), dim=1).to(self.device)

            # SAMPLE API Positions. Each data point can have up to k_positions API calls
            sampled_positions, current_values = self.sample_API_positions(
                model=model,
                prompt_data_pad=prompt_data_pad,
                prompt_size = prompt_size,
                data_lengths=data_length,
                sampling_threshold=0,
            )

            # positions and values
            print(torch.cat([sampled_positions.unsqueeze(0), current_values.unsqueeze(0)], dim=0).view(2,-1).T)
            # Convert sampled_positions to 1-dimensional vector
            
            print("Current values:", flush=True)
            print(current_values.view(-1).tolist())
            logging.debug(f"Current values: {current_values.view(-1).tolist()}")

            values += current_values.view(-1).tolist()
            lengths += data_length.tolist()

        return values, lengths



    @torch.no_grad()
    def forward(
        self,
        data: List[str],
        additional_input=None
    ):
        
        print("FORWARD", flush=True)
        logging.info(f"FORWARD")
        self.model.eval()
        start_total_time = time.time()
        print(f"Received a batch of size {len(data)}", flush=True)

        new_to_old_idx = [i for i in range(len(data))]

        # We load the data substituted into the prompt so that the model samples appropriate positions for API calls
        dataset = PromptDataset(
            data=data,
            tokenized_raw_prompt=self.tokenized_raw_prompt,
            tokenized_arg_prompt=self.tokenized_raw_arg_prompt,
            substitute_index=self.substitute_index,
            tokenizer_encode=self.encode,
            new_to_old_idx=new_to_old_idx,
        )
        dl = PromptDataloader(
            dataset,
            batch_size=self.prompt_batch_size
        )

        tokenized_data = []
        additional_output = {}    # This dictionary is indexed with a call index and contains additional information for posterior analysis
        generated_calls = []
        call_counter = 0          # This counter indexes calls.
        batch0_counter = 0
        arg_gen_batch_size = self.prompt_batch_size
        split_batch = False       # Split position sampling batch in two if size is too big for GPU
        errors_left = self.MAX_CUDA_ERROR_COUNT

        #generation_stats['Time to generate dataloader'] = prev_time - start
        #logging.info(f"PromptDataloader took {time.time() - prev_time} seconds")

        prev_time = time.time()
        timers = {
            "pos":0,
            "pos_post":0,
            "arg":0,
            "duplicates":0,
        }
        # Dictionary for additional stats of each part of the pipeline
        generation_stats = {
            'Number of input examples': 0,
            'Number of sampled positions': 0,
            'Number of sampled arguments': 0,
            'Number of non-duplicate arguments': 0,
            'Average data length': 0,
            'Max data length': 0,
            'Min data length': TRUNCATE_LENGTH
        }
        for part in ["prepare", "forward", "gumbel sample", "postprocess and stopping"]:
            generation_stats[f'Time to sample arguments - part "{part}" (avg/example)'] = 0

        # Iterate dataset (batch0)
        for inserted_prompt_list, data_list, data_length, data_indices in dl:
            # Data length stats:
            generation_stats['Average data length'] += data_length.sum().item()
            generation_stats['Max data length'] = max(generation_stats['Max data length'], data_length.max().item())
            generation_stats['Min data length'] = min(generation_stats['Min data length'], data_length.min().item())
            start = time.time()
            start_batch_time = time.time()

            tokenized_data += data_list     # Keep list of tokenized data and lengths

            prompts = left_pad_sequence(inserted_prompt_list, padding_value=PAD_ID).to(self.device)
            data_pad = pad_sequence(data_list, padding_value=PAD_ID).to(self.device)
            prompt_data_pad = torch.cat((prompts, data_pad), dim=1)

            if not split_batch:
                try:
                    # SAMPLE API Positions. Each data point can have up to k_positions API calls
                    sampled_positions, max_call_count = self.sample_API_positions(
                        prompt_data_pad=prompt_data_pad,
                        prompt_size = prompts.shape[1],
                        data_lengths=data_length,
                        sampling_threshold=self.pos_threshold,
                    )
                    
                except torch.cuda.OutOfMemoryError as oome: # type: ignore
                    torch.cuda.empty_cache()
                    logging.info(oome)
                    logging.info(f"{arg_gen_batch_size} was too large and caused an OOM exception.")
                    logging.info("Splitting positions in two batches.")
                    # Data lengths in increasing order. As soon as there is a OOM exception, all subsequent batches will also cause an OOM exception.
                    split_batch = True
            if split_batch:
                sampled_positions1, max_call_count1 = self.sample_API_positions(
                    prompt_data_pad=prompt_data_pad[:self.prompt_batch_size//2],
                    prompt_size = prompts.shape[1],
                    data_lengths=data_length[:self.prompt_batch_size//2],
                )
                if self.prompt_batch_size//2 < data_length.shape[0]-1:
                    sampled_positions2, max_call_count2 = self.sample_API_positions(
                        prompt_data_pad=prompt_data_pad[self.prompt_batch_size//2:],
                        prompt_size = prompts.shape[1],
                        data_lengths=data_length[self.prompt_batch_size//2:],
                    )
                else:
                    sampled_positions2 = torch.tensor([], dtype=torch.long, device=self.device)
                    max_call_count2 = 0
                sampled_positions = torch.cat([sampled_positions1, sampled_positions2], dim=0)
                max_call_count = max(max_call_count1, max_call_count2)


            if call_counter == 0 and DEVICE.type == 'cuda':
                print(torch.cuda.memory_summary(device=self.device))
                logging.debug("CUDA SUMMARY AFTER POSITION SAMPLING")
                logging.debug(torch.cuda.memory_summary(device=self.device))
                torch.cuda.reset_max_memory_allocated(device=self.device)

            timers["pos"] += time.time() - start
            start = time.time()
            if max_call_count == 0:
                print("No API positions were above threshold")
                logging.info("No API positions were above threshold")
                continue

            if self.debug_level > 1:
                logging.debug("SAMPLED POSITIONS")
                logging.debug(f"Shape: {sampled_positions.shape}")
                logging.debug(sampled_positions)
            
            indices = sampled_positions.nonzero()[:,0].squeeze(0)  # Data index (row indices) of each sampled position-point in vector format
            positions = sampled_positions.masked_select(sampled_positions != 0) # Boolean tensor of useful positions
            

            print(f"indices {indices}")
            print(f"Positions {positions}")
            idx_pos = torch.stack([indices, positions], dim=1)    # Tuples of indices and positions
            logging.info(f"Index pos tuples will be interleaved {self.m_arg_samples} times")
            logging.info(idx_pos.T)
            idx_pos = idx_pos.repeat_interleave(self.m_arg_samples, dim=0)  # Repeat each entry m_arg_times, for each position we sample m_arg_samples arguments:
            n_samples = idx_pos.shape[0]   # Number of samples to compute

            # STATS --------------------------------------------------------------
            generation_stats["Number of input examples"] += data_length.shape[0]
            generation_stats['Number of sampled positions'] += indices.shape[0]
            timers["pos_post"] += time.time() - start
            start = time.time()
            average_arg_timers = {
                "prepare":0,
                "forward":0,
                "gumbel sample":0,
                "postprocess and stopping":0,
            }
            batch0_gen_args_count = 0

            if MASKED_ARG_GENERATION:
                idx_pos_sort = torch.argsort(idx_pos[:, 1])
                arg_gen_prompts = [self.tokenized_raw_arg_prompt]*len(inserted_prompt_list)
            else:
                idx_pos_sort = torch.argsort(idx_pos[:, 1] +  data_length[idx_pos[:, 0].cpu()],)
                arg_gen_prompts = inserted_prompt_list
            idx_pos = idx_pos[idx_pos_sort]
            prompt_data_pad = []
            for i, pos in idx_pos[:arg_gen_batch_size]:
                prompt_data_pad.append(torch.cat((Tensor(arg_gen_prompts[i]), data_list[i][:pos])).to(self.device))
            prompt_data_pad = left_pad_sequence(prompt_data_pad, padding_value=PAD_ID)


            calls = {}
            # Split positions into batches and sample arguments (batch1)
            count_idx_pos_processed = 0
            pos_batch_i = 0
            error_count = 0
            while count_idx_pos_processed < n_samples:
                batch1_args_count = 0
                try:
                    logging.debug(f"{pos_batch_i}th iteration. Processed {count_idx_pos_processed} samples which is {count_idx_pos_processed/n_samples*100}%.")
                    if self.debug_level > 1: 
                        logging.debug(f"Indices: {idx_pos[:arg_gen_batch_size,0]}")
                        logging.debug(f"Positions: {idx_pos[:arg_gen_batch_size,1]}")
                    # Output is a dict similar to the calls dict
                    output, call_counter, part_timers = self.generate_API_call_arguments(
                        prompt_data=prompt_data_pad[:arg_gen_batch_size],
                        positions=idx_pos[:arg_gen_batch_size,1],
                        data_indices=data_indices[idx_pos[:arg_gen_batch_size,0]],
                        call_counter=call_counter,
                        temperature=self.sampling_temperature,
                        random_finish_p = 1.0 if self.max_arg_length == 0 else 0.0,
                    )
                except torch.cuda.OutOfMemoryError as oome: # type: ignore
                    torch.cuda.empty_cache()
                    error_count += 1
                    logging.info(oome)
                    logging.info(f"{arg_gen_batch_size} was too large and caused an OOM exception.")
                    logging.info(f"Decreasing batch size by 5 and trying again.")
                    logging.info(f"Number of fails: {error_count}")
                    arg_gen_batch_size -= self.BATCH_SIZE_DECREASE_STEP
                    if error_count < self.MAX_CUDA_ERROR_COUNT:
                        continue
                    else:
                        logging.info(f"Number of fails exceeded {self.MAX_CUDA_ERROR_COUNT}. Aborting.")
                        raise oome
                    
                idx_pos = idx_pos[arg_gen_batch_size:]
                prompt_data_pad = []
                for i, pos in idx_pos[:arg_gen_batch_size]:
                    prompt_data_pad.append(torch.cat((Tensor(arg_gen_prompts[i]), data_list[i][:pos])).to(self.device))
                if len(prompt_data_pad) > 0:
                    prompt_data_pad = left_pad_sequence(prompt_data_pad, padding_value=PAD_ID)

                for part, time_taken in part_timers.items():
                    average_arg_timers[part] = average_arg_timers[part] * pos_batch_i + time_taken/min(arg_gen_batch_size, n_samples - count_idx_pos_processed)
                    average_arg_timers[part] /= pos_batch_i + 1

                for index, call in output.items():
                    calls[str(index)] = calls.get(str(index), []) + call
                    batch1_args_count += len(call)
                    if self.debug_level > 1:
                        for example in call:
                            pos = example.position
                            idx = example.data_idx
                            e_len = tokenized_data[idx].shape[-1]
                            logging.debug(f"""Call {example.call_idx} pos {pos}: {DECODE(tokenized_data[idx][:pos]) + " [" + self.tool_name + "(" + DECODE(example.tokenized_args.get()) + ")] " + DECODE(tokenized_data[idx][pos:])}""")
                            logging.debug(f"With position {pos} and data index {idx}. Token at pos-2:{DECODE(tokenized_data[idx][pos-2]) if pos>1 else ''}, pos-1: {DECODE(tokenized_data[idx][pos-1]) if pos>0 else ''}, pos: {DECODE(tokenized_data[idx][pos])}, pos+1: {DECODE(tokenized_data[idx][pos+1]) if pos+1 < e_len else ''}, pos+2: {DECODE(tokenized_data[idx][pos+2]) if pos+2<e_len else ''}")
                
                logging.debug("-----------------------------------")
                logging.debug(f"Batch {pos_batch_i}:  Generated {batch1_args_count} calls out of {positions.shape[0]} positions")
                logging.debug("-----------------------------------")
                
                batch0_gen_args_count += batch1_args_count  # Reset each batch0 iteration
                count_idx_pos_processed += min(arg_gen_batch_size, n_samples - count_idx_pos_processed)  
                pos_batch_i += 1

            logging.debug("\n==================================================================")
            logging.info(f"SAMPLED {batch0_gen_args_count} ARGUMETS OF BATCH {batch0_counter} IN {time.time() - start_batch_time}s:")
            logging.info(f"Checking for duplicates")

            batch0_counter += 1
            generation_stats['Number of sampled arguments'] += batch0_gen_args_count
            for key, value in average_arg_timers.items():
                generation_stats[f'Time to sample arguments - part "{key}" (avg/example)'] *= batch0_counter
                generation_stats[f'Time to sample arguments - part "{key}" (avg/example)'] += value
                generation_stats[f'Time to sample arguments - part "{key}" (avg/example)'] /= batch0_counter + 1

            timers["arg"] += time.time() - start
            start = time.time()

            # Check for call duplicates
            valid_count = 0
            for index, call_list in calls.items():
                if call_list is None or call_list == []:
                    logging.warn(f"Call list for index {index} is empty")
                    continue
                
                decoded_args = []
                for call in call_list:
                    args = DECODE(call.tokenized_args.get())
                    if self.preprocess_args:
                        # Gather some additional data for analysis
                        extra_data = additional_output.get(call.call_idx, {})
                        extra_data['raw_arg'] = args

                        # Preprocess args
                        args = self.preprocess_args(args)

                        # Gather some additional data for analysis
                        extra_data['processed_arg'] = args
                        additional_output[call.call_idx] = extra_data
                        call.tokenized_args.update(Tensor(self.encode(args)).long())
                    decoded_args.append(args)

                valid_indices = self.tool_check_duplicates(decoded_args)

                for valid_index in valid_indices:
                    valid_call = call_list[valid_index]
                    if len(valid_indices) < len(call_list):
                        extra_data = additional_output.get(valid_call.call_idx, {})
                        extra_data['arg_cohort'] = [DECODE(call.tokenized_args.get()) for call in call_list]
                        additional_output[valid_call.call_idx] = extra_data
                    generated_calls.append(call_list[valid_index])
                    valid_count += 1
            logging.debug("*****************************************************************")
            logging.info(f"{valid_count} calls out of {batch0_gen_args_count} were valid")

            generation_stats['Number of non-duplicate arguments'] += valid_count
            timers["duplicates"] += time.time() - start

                #print(f"Call: {DECODE(t[0])}")
            # SAMPLE API ARGS

        generation_stats['Time to sample (TOTAL)'] = time.time() - prev_time
        generation_stats['Time to sample positions'] = timers['pos']
        generation_stats['Time to sample pos post'] = timers['pos_post']
        generation_stats['Time to sample arguments'] = timers['arg']
        generation_stats['Time to check duplicates'] = timers['duplicates']
        generation_stats['Average data length'] /= len(tokenized_data)
        prev_time = time.time()

        logging.info("\n" + "="*66 + "\n\n" + "="*66)
        logging.info(f"SAMPLED {len(generated_calls)} CALLS IN {time.time() - prev_time}s")
        logging.info(f"Pos took {timers['pos'] + timers['pos_post']}s ({timers['pos_post']}s post pos)   //   Arg took {timers['arg']}s   //  Duplicates took {timers['duplicates']}s")        
        print(f"SAMPLED {len(generated_calls)} CALLS IN {time.time() - prev_time}s", flush=True)
        print(f"Pos took {timers['pos'] + timers['pos_post']}s ({timers['pos_post']}s post pos)   //   Arg took {timers['arg']}s   //  Duplicates took {timers['duplicates']}s")
        print(flush=True)

        #discarded_data = []
        #for i in discarded_indices:
            #discarded_data.append(data_list.pop(i))

        # TODO add to discarded data useless data if generation bad

        calls_and_responses = []
        logging.info("EXECUTING API CALLS")
        for call in generated_calls:
            tokenized_args = call.tokenized_args.get()
            args = self.decode(tokenized_args)
            if self.debug_level>1: logging.debug(f"Call: {call.call_idx} args: {args}")
            try:
                parenthesis_idx = args.rindex(')')
                logging.debug("Args with parenthesis?")
                print("Args with parenthesis?")
                print(args)
                args = args[:parenthesis_idx]
                print(args)
            except ValueError as e:
                True
                #print(f"args \"{args}\" did not have closing parenthesis")
            
            response = None
            try:
                argv = additional_input[new_to_old_idx[call.data_idx]] if additional_input else None
                response = self.tool(args, argv)
                if isinstance(response, float): response = str(response)
                assert isinstance(response, str), f"API response has type {type(response)} is not a string"

                response = self.encode(response, truncation=True, max_length=TRUNCATE_LENGTH, return_tensors='pt').long().squeeze(0)
                call_len = len(response) + len(tokenized_args) + len(tokenized_data[call.data_idx])
                
                calls_and_responses.append((tokenized_args, response, call.position, call.data_idx, call.call_idx, call_len))
            except Exception as e:
                logging.debug(f"args {args} caused tool to fail")
                logging.debug(f"Caught exception: {e}")
                print(f"args {args} caused tool to fail")
                print(f"Caught exception: {e}")
                
        logging.debug(f"\n==================================================================\n\n=======================================================================")
        logging.info(f"EXECUTED {len(calls_and_responses)} API calls OUT OF {len(generated_calls)} IN {time.time() - prev_time}s:")
        print()
        
        generation_stats['Number of executed calls'] = len(calls_and_responses)
        generation_stats['Time to execute calls'] = time.time() - prev_time
        prev_time = time.time()

        if DEVICE.type == 'cuda':
            # Cuda memory sumary:
            print(torch.cuda.memory_summary(device=self.device))
            logging.info(torch.cuda.memory_summary(device=self.device))
            torch.cuda.reset_max_memory_allocated(device=self.device)

        # We have to sort the calls_and_responses by response length
        calls_and_responses.sort(key=lambda x: x[-1])

        #calls_and_responses *= 100   # TODO REMOVE

        logging.debug(f"LEN OF CALLS AND RESPONSES: {len(calls_and_responses)}")

        extended_dataset = APICallDataset(
            data=tokenized_data,
            call_response_pos_idxs=calls_and_responses,
            tokenizer_encode=self.encode,
            tokenizer_decode=self.decode,
            device=self.device,
            tool_name=self.tool_name
        )

        dl_API = APIDataloader(
            extended_dataset,
            batch_size=self.prompt_batch_size
        )
        data_iter = iter(dl_API)

        #generation_stats['API_calls_dataset'] = time.time() - prev_time
        #logging.info(f"API Dataloader took {time.time() - prev_time} seconds")
        prev_time = time.time()

        # Filter by perplexity improvement
        num_passed = 0
        num_failed = 0
        results = []

        logging.info("FILTERING API CALLS")


        filter_batch_size_target = self.filtering_batch_size
        print(f"Filtering batch size target: {filter_batch_size_target}")
        logging.info(f"Filtering batch size target: {filter_batch_size_target}")

        pad = partial(pad_sequence, batch_first=True, padding_value=PAD_ID)

        def new_batch(data = None):
            if data is None:
                dummy_t = Tensor([]).long()
                batch = {
                    "data_triplet":[[], [], []],
                    "position_triplet":[dummy_t, dummy_t, dummy_t],
                    "clean_calls":[],
                    "clean_resps":[],
                    "data_indices":[],
                    "call_indices":[],
                    "length":0,
                }
            else:
                batch = {
                    "data_triplet":[list(data[0][0]), list(data[0][1]), list(data[0][2])],
                    "position_triplet":[data[1][0], data[1][1], data[1][2]],
                    "clean_calls":data[2][0],
                    "clean_resps":data[2][1],
                    "data_indices":list(data[3]),
                    "call_indices":list(data[4]),
                    "length":len(data[0][0]),
                }
            return batch
            
        # Pending is a list of batches. If the batch is not full, we split the next batch and fill the current batch.
        pending = new_batch()
        spare_data = []   # Store leftover data from previous batch
        # Custom wrapper collage function, to cut and paste dataloader output to enable dynamic batch size
        def fill_pending(data=None):
            nonlocal pending, spare_data
            # Batch to fill to target size

            while pending["length"] < filter_batch_size_target:
                if data is None:
                    if len(spare_data) > 0:
                        data = spare_data.pop(-1)
                    else:
                        try:
                            data = new_batch(next(data_iter))
                        except StopIteration:
                            return
                missing_n = filter_batch_size_target - pending["length"]
                leftover = new_batch()

                assert len(data["data_triplet"][0]) == data["position_triplet"][0].shape[0], "Tuple of data and positions should have same length"
                
                for i in range(3):
                    # Data_triplet is a tuple of 3 tensors: data, call, response
                    pending["data_triplet"][i] += data["data_triplet"][i][:missing_n]
                    leftover["data_triplet"][i] = data["data_triplet"][i][missing_n:]	

                    pending["position_triplet"][i] = torch.cat([pending["position_triplet"][i], data["position_triplet"][i][:missing_n]], dim=0)
                    leftover["position_triplet"][i] = data["position_triplet"][i][missing_n:]
                
                pending["clean_calls"].extend(data["clean_calls"][:missing_n])
                leftover["clean_calls"].extend(data["clean_calls"][missing_n:])
                pending["clean_resps"].extend(data["clean_resps"][:missing_n])
                leftover["clean_resps"].extend(data["clean_resps"][missing_n:])

                pending["data_indices"].extend(data["data_indices"][:missing_n])
                leftover["data_indices"].extend(data["data_indices"][missing_n:])
                pending["call_indices"].extend(data["call_indices"][:missing_n])
                leftover["call_indices"].extend(data["call_indices"][missing_n:])
                
                pending["length"] += min(data["length"], missing_n)
                leftover["length"] = max(data["length"] - missing_n, 0)
                """
                logging.debug("FILLING PENDING BATCH")
                logging.debug(f"Batch: {batch}")
                logging.debug(batch == pending[-1])
                logging.debug(f"Pending len {len(pending)}")
                for pend in pending:
                    logging.debug(f"Len of pend: {pend['length']}")
                logging.debug("LENG|THS:")
                logging.debug(f"Old batch: {batch['length']-data['length']}")
                logging.debug(f"leftover len: {leftover['length']}")
                logging.debug(f"Data : {data['length']}")
                logging.debug(f"New batch: {batch['length']}")"""

                if data["length"] - missing_n > 0:
                    spare_data.append(leftover)

                data = None

                print(f"Pending length {pending['length']}")
                print(f"Leftover length {leftover['length']}")

              
        fill_pending()
        while pending["length"] > 0:

            if self.debug_level > 1:
                logging.debug(f"Pending: {pending}")

            data_triplet = pending["data_triplet"]
            # Pad data_triplet to same length
            pad = partial(pad_sequence, batch_first=True, padding_value=PAD_ID)
            padded_data_triplet = tuple([pad(x) for x in data_triplet])
            if LLAMA:
                # Add bos token to the data_triplet:
                data_tokens, data_call_tokens, data_resp_tokens = [torch.cat([BOS_ID*torch.ones((x.shape[0],1)), x.cpu()], dim=1).long() for x in padded_data_triplet]

            position_triplet = pending["position_triplet"]
            #clean_calls, clean_resps), data_indices, call_indices = pending
            clean_calls = pending["clean_calls"]
            clean_resps = pending["clean_resps"]
            data_indices = pending["data_indices"]
            call_indices = pending["call_indices"]
            logging.info(f"Batch size: {pending['length']}")
            logging.info("Data example:")
            # Print data_indices, call_indices, and position_triplet of first example
            logging.info(f"Data indices: {data_indices[0]}, call indices: {call_indices[0]}, positions: {position_triplet[0][0]}")
            logging.info(f"Data: {DECODE(data_tokens[0])}")
            logging.info(f"API call: {DECODE(data_call_tokens[0])}")
            logging.info(f"API call response: {DECODE(data_resp_tokens[0])}")
            # print original data point
            logging.info(f"Original data: {DECODE(tokenized_data[data_indices[0]])}")

            try:
                logging.debug(f"SIZE OF DATA: {data_tokens.shape}")
                logging.debug(f"LENGTH OF BATCH: {pending['length']}")
                print(f"SIZE OF DATA: {data_tokens.shape}")
                print(f"LENGTH OF BATCH: {pending['length']}")
                output = self.filter_tokens_with_api_response(
                    data_tokens, 
                    data_call_tokens, 
                    data_resp_tokens, 
                    position_triplet,
                    Tensor(data_indices), 
                    Tensor(call_indices), 
                    filter_threshold=self.filter_threshold, 
                    weighting_fn=self.weight_func
                )
            except torch.cuda.OutOfMemoryError as oome: # type: ignore
                torch.cuda.empty_cache()
                logging.info(oome)
                logging.info(f"{filter_batch_size_target} was too large and caused an OOM exception.")
                filter_batch_size_target -= self.BATCH_SIZE_DECREASE_STEP*17
                logging.info(f"Reducing size to {filter_batch_size_target} and trying again.")
                # Data lengths in increasing order. As soon as there is a OOM exception, all subsequent batches will also cause an OOM exception.
                if filter_batch_size_target < 1:
                    raise oome

                # Reduce batch size and try again
                old_pending = pending
                pending = new_batch()
                fill_pending(old_pending)
                continue
            
            num_passed += output['num_passed']
            num_failed += output['num_failed']
            
            selected_mask = output['selected_mask']
            selected_indices = selected_mask.nonzero().squeeze(1).tolist()
            selected_calls = [clean_calls[i] for i in selected_indices]
            selected_responses = [clean_resps[i] for i in selected_indices]

            for index, loss, call, resp, pos, call_idx in zip(output['selected_indices'], output['loss_selected'], selected_calls, selected_responses, position_triplet[0][selected_mask], output['call_indices']):
                if LLAMA: pos = pos - 1   # Observed in LLama pos is one off, perhaps <bos>?
                call_idx = call_idx.item()
                tokenized_text = tokenized_data[index]

                # TODO save as tensors or lists or what?
                text_call = torch.cat([tokenized_text[:pos], call, tokenized_text[pos:]])   
                text_resp = torch.cat([tokenized_text[:pos], resp, tokenized_text[pos:]])   

                # TODO add positions
                row = {'text': DECODE(tokenized_text), 'API_calls_text': DECODE(text_call), 
                       'API_call_response_text': DECODE(text_resp), 'positon': pos.item(), 
                       'loss_improvement': loss.item(), 
                       'index': new_to_old_idx[index]}
                logging.debug(f"Saving call {call_idx} with loss {loss.item()}")
                logging.debug(f"text: {row['text']}")
                logging.debug(f"text1: {row['API_calls_text']}")
                logging.debug(f"text2: {row['API_call_response_text']}")
                logging.debug(f"Call {call_idx} pos {pos}")
                
                if len(additional_output.get(call_idx, {})) > 0:
                    for key, value in additional_output[call_idx].items():
                        row[key] = value
                results.append(row)

            pending = new_batch()
            fill_pending() 

        # STATS
        generation_stats['Time to filter'] = time.time() - prev_time
        generation_stats['Number of calls passed filtering'] = num_passed
        generation_stats['Time (total pipeline)'] = time.time() - start_total_time
        generation_stats['Time key to example length for per example averaging'] = {
            'Time to sample (TOTAL)': generation_stats['Number of input examples'],
            'Time to sample positions': generation_stats['Number of input examples'],
            'Time to sample pos post': generation_stats['Number of sampled positions'],
            'Time to sample arguments': generation_stats['Number of sampled positions']*self.m_arg_samples,
            'Time to check duplicates': generation_stats['Number of sampled arguments'],
            'Time to execute calls': generation_stats['Number of non-duplicate arguments'],
            'Time to filter': generation_stats['Number of executed calls'],
            'Time (total pipeline)': generation_stats['Number of input examples'],
        }

        # LOGS
        print(f"FILTERED API calls in {time.time() - prev_time}s")
        print(generation_stats)
        logging.info(f"FILTERED API calls in {time.time() - prev_time}s")
        logging.info(f"num passed: {num_passed}")
        logging.info(f"num failed: {num_failed}")
        for key, value in generation_stats.items():
            logging.info(f"{key}: {value}")
        logging.info("*"*66 + "\n\n" + "*"*66 + "\n\n" + "*"*66)
        if DEVICE.type == 'cuda':
            # Cuda memory sumary:
            print(torch.cuda.memory_summary(device=self.device))
            logging.info(torch.cuda.memory_summary(device=self.device))

        return results, generation_stats



#########################################################################################################
#########################################################################################################