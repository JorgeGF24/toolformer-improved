from transformers import GPT2Tokenizer, GPT2LMHeadModel,AutoTokenizer, AutoModelForCausalLM
from toolformer import Toolformer
import torch

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, cache_dir=cache_dir
    ).cuda()

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
#model = GPT2LMHeadModel.from_pretrained('gpt2-medium')


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()

    
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1, eps=1e-10):
    if temperature == 0:
        return t.argmax(dim=dim)
        
    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim=dim)




def predictions(prime):

    test_string = [prime]

    test_tokens = torch.tensor(tokenizer.encode(test_string[0]).append(102), device="cuda").long()

    test_logits = model(test_tokens).logits
    _, top_logits = torch.topk(test_logits, k=5, dim=-1)

    print(top_logits.shape)
    test_sample = gumbel_sample(test_logits, temperature=0)

    print("MODEL OUTPUT")

    for i, token in enumerate(test_sample):
        print(f"{i}: {tokenizer.decode(token)} // {tokenizer.decode(test_tokens[i+1])}")
        print(f"Top predictions: {tokenizer.decode(top_logits[i])}")


prime = """Your task is to add calls to a Calendar API to a piece of text.
    The API calls should help you get information required to complete the text.
    You can call the API by writing "[Calendar()]".
    Here are some examples of API calls:
    Input: Today is the first Friday of the year.
    Output: Today is the first [Calendar()] Friday of the year.
    Input: The president of the United States is Joe Biden.
    Output: The president of the United States is [Calendar()] Joe Biden.
    Input: The store is never open during the week, so today it is closed.
    Output:"""

predictions(prime)