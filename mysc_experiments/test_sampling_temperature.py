from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("STARTING", flush=True)

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, cache_dir=cache_dir
).cuda()

print("STF")

def log(t, eps=1e-20): return t.clamp(min=eps).log()


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1, eps=1e-10):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim=dim)


model.eval()

text = "The cat sat on the mat"
logits = model(torch.tensor(tokenizer.encode(), device="cuda").unsqueeze(0)).logits

# Sample from the logits by taking the max arg

sampled_token_no_rand = gumbel_sample(logits, temperature=0.5)

print(tokenizer.decode(sampled_token_no_rand[0]))

for temp in range(1, 10):
    for i in range(10):
        logits = model(torch.tensor(tokenizer.encode(), device="cuda").unsqueeze(0)).logits
        sampled_token = gumbel_sample(logits, temperature=0.6 + temp/10)
        print(f"{i}th Temperature: {0.6 + temp/10}")
        print(tokenizer.decode(sampled_token[0]))