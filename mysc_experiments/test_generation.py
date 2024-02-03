# Test model generation

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

"""
cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
cache_option = {"cache_dir": cache_dir}
tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, **cache_option)

tokenizer1.add_tokens(["[PAD]"])
tokenizer1.pad_token="[PAD]"

model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, cache_dir=cache_dir).cuda()


tokenizer2 = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf",
                                            token="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",
                                            **cache_option)

tokenizer2.add_bos_token = False

tokenizer2.add_tokens(["[PAD]"])
tokenizer2.pad_token="[PAD]"

model2 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                            token="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",
                                            **cache_option)"""



# Test generation
from transformers import pipeline

generator = pipeline(model="EleutherAI/gpt-j-6B")
prompt = "Marilyn's first record sold 10 times as many copies as Harald's. If they sold 88,000 copies combined, how many copies did Harald sell?"

print(generator(prompt, do_sample=False))



"""
from transformers import TextGenerationPipeline

generator = TextGenerationPipeline(model, tokenizer1, device=0)

prompt = "Marilyn's first record sold 10 times as many copies as Harald's. If they sold 88,000 copies combined, how many copies did Harald sell?"

print(generator(prompt, max_length=100, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, repetition_penalty=1.2, num_return_sequences=5))

print("Finished generating")"""

