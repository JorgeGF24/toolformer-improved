
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPTJConfig, LlamaConfig
from toolformer import Toolformer
import torch



cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
cache_option = {"cache_dir": cache_dir} if cache_dir else {} 


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





toolformer = Toolformer(
        model=model,
        pad_token = tokenizer.pad_token, 
        max_arg_length=30,
        max_data_length=130,
        max_response_length=100,
        m_arg_samples=5,
        prompt_batch_size=100,
        filtering_batch_size=100,
        raw_tool_prompt="HELLO",
        tool_name="Calculator",
        tool=lambda x: x,
        tool_check_duplicates=lambda x: x,
        tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode,
        log_dir="./logs-filtering",
        debug_level=2,
        arg_gen_stoppers=torch.Tensor(1),
        experiment_config={},
        using_llama= True,
        api_start_token = "["
    ).cuda()

text = ["Similarly osculator for 49 is 5"," (taken from 5×10 = 50)."]
text_call = ["Similarly osculator for 49 is 5 [Calculator((49/5))]"," (taken from 5×10 = 50)."]
text_response = ["Similarly osculator for 49 is 5 [Calculator((49/5))→ 9.8]"," (taken from 5×10 = 50)."]

tokenized_text = [tokenizer.encode(t) for t in text]
tokenized_text_call = [tokenizer.encode(t) for t in text_call]
tokenized_text_response = [tokenizer.encode(t) for t in text_response]

positions = [torch.tensor(len(t[0])) for t in (tokenized_text, tokenized_text_call, tokenized_text_response)]

tokenized_text = torch.tensor(tokenized_text[0] + tokenized_text[1]).unsqueeze(0).cuda()
tokenized_text_call = torch.tensor(tokenized_text_call[0] + tokenized_text_call[1]).unsqueeze(0).cuda()
tokenized_text_response = torch.tensor(tokenized_text_response[0] + tokenized_text_response[1]).unsqueeze(0).cuda()

data_indices = torch.tensor(list(range(tokenized_text.shape[0]))).unsqueeze(0).cuda()

toolformer.filter_tokens_with_api_response(
    tokens=tokenized_text,
    tokens_without_api_response=tokenized_text_call,
    tokens_with_api_response=tokenized_text_response,
    data_indices=data_indices,
    filter_position=positions,
)