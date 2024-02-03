# Test the generation of the GPT-J model

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import torch

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
cache_option = {"cache_dir": cache_dir} if cache_dir else {}

tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/tokenizer", truncate=True, max_length=270, **cache_option)

# Empty cuda cache
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, **cache_option
).cuda()

model.eval()

model.resize_token_embeddings(len(tokenizer))

print("Model loaded")

prompt = """Your task is to add calls to a Calculator API to a piece of text. The calls should help you get information required to complete the text. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of API calls:
Input: Last year we collected 237342 apples,double of what we collected this year: 118671.
Output: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→118671] 118671.
Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)→54] 54.
Input: A total of 252 matches were played, and 723 goals were scored (an average of 2.87 per match). This is twenty goals more than the 703 goals last year.
Output: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)→2.87] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)→17] 17 years.
Input: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of 77.7.
Output: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of [Calculator("""

prompt_len = len(tokenizer.encode("""Your task is to add calls to a Calculator API to a piece of text. The calls should help you get information required to complete the text. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of API calls:
Input: Last year we collected 237342 apples, double of what we collected this year: 118671.
Output: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→118671] 118671.
Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)→54] 54.
Input: A total of 252 matches were played, and 723 goals were scored (an average of 2.87 per match). This is twenty goals more than the 703 goals last year.
Output: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)→2.87] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)→17] 17 years.
Input: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of 77.7.
Output: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of"""))

config = GenerationConfig(
    max_new_tokens=10,
    num_beams=25,
    early_stopping=True,
    #top_k=50,
    #top_p=0.95,
    num_return_sequences=25,
    pad_token_id=tokenizer.pad_token_id,
)


output = model.generate(
    inputs= tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=config,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
)

# move output to cpu
output = output.cpu()

"""
config2 = GenerationConfig(
    max_new_tokens=10,
    do_sample=True,
    top_k=50,
    #top_p=0.95,
    temperature=1.0,
    num_return_sequences=25,
    pad_token_id=tokenizer.pad_token_id,
    )


output2 = model.generate(
    inputs= tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=config,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
)

# move output to cpu
output2 = output2.cpu()

config3 = GenerationConfig(
    max_new_tokens=12,
    num_beams=1,
    do_sample=True,
    #top_k=50,
    top_p=0.6,
    temperature=1.0,
    num_return_sequences=25,
    pad_token_id=tokenizer.pad_token_id,
    )


output3 = model.generate(
    inputs= tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=config,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
)

# move output to cpu
output3 = output3.cpu()"""


# decode
print("________________________ OUTPUT 1 ________________________")
for sentence in output:
    print(tokenizer.decode(sentence[prompt_len:]))
    print("")
"""
print("________________________ OUTPUT 2 ________________________")
for sentence in output2:
    print(tokenizer.decode(sentence[prompt_len:]))
    print("")

print("________________________ OUTPUT 3 ________________________")
for sentence in output3:
    print(tokenizer.decode(sentence[prompt_len:]))
    print("")

test_attention = ["73, 73, 73, 73",
                "Random number: "]

inputs = tokenizer.batch_encode_plus(test_attention, return_tensors="pt", padding=True).input_ids.cuda()


output = model.generate(
    inputs= inputs,
    generation_config=config,
)


# decode
for sentence in output:
    print(tokenizer.decode(sentence))
    print("")"""


print(torch.cuda.memory_summary(device=model.device))
