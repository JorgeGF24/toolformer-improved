from csv import DictWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
from toolformer import Toolformer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

if __name__ == "__main__":

    cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
    tokenizer.add_tokens('[PAD]')
    tokenizer.pad_token = '[PAD]'

    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, cache_dir=cache_dir
        ).cuda()

    model.resize_token_embeddings(len(tokenizer))

    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    #model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    # simple calendar api call - function that returns a string


    def Calendar(arg = "", date = None ):
        assert len(arg) == 0 or arg == " ", "Argument to the Calendar API should be empty."
        import datetime
        from calendar import day_name, month_name
        # datetime from date that is a string from a datetime object:
        if date is not None:
            now = datetime.datetime.strptime(date, '%Y-%m-%d')
        else:
            now = datetime.datetime.now()
        return f'Today is {day_name[now.weekday()]}, {month_name[now.month]} {now.day}, {now.year}'

    # prompt for teaching it to use the Calendar function from above


    prompt = f"""Your task is to add calls to a Calendar API to a piece of text.
    The API calls should help you get information required to complete the text.
    You can call the API by writing "[Calendar( )]".
    Here are some examples of API calls:
    Input: Today is the first Friday of the year.
    Output: Today is the first [Calendar( )] Friday of the year.
    Input: The president of the United States is Joe Biden.
    Output: The president of the United States is [Calendar( )] Joe Biden.
    Input: [PAD]
    Output: """


    encoded_prompt = torch.tensor(tokenizer.encode(prompt)).long().cuda()
    output = model(encoded_prompt)

    data = [
        "The store is only open during the weekend, so today it is closed.",
        "The number of days from now until Christmas is 180.",
        "The current day of the week is Wednesday.",
        "The current day of the month is 28.",
        "The current year is 2023."
    ]


    def Calculator(input_query: str):
        operators = {
            '+': add,
            '-': sub,
            '*': mul,
            '/': truediv
            }
        if input_query.isdigit():
            return float(input_query)
        for c in operators.keys():
            left, operator, right = input_query.partition(c)
            if operator in operators:
                return round(operators[operator](Calculator(left), Calculator(right)), 2)

    # wolfarm_alpha_appid is WVAAYJ-U484V3WVG5
    def WolframAlphaCalculator(input_query: str):
        wolfram_alpha_appid = WVAAYJ-U484V3WVG5 #os.environ.get('WOLFRAM_ALPHA_APPID')
        wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
        res = wolfram_client.query(input_query)
        assumption = next(res.pods).text
        answer = next(res.results).text
        return f'Assumption: {assumption} \nAnswer: {answer}'

    calculator_prompt = f"""Your task is to add calls to a Calculator API to a piece of text.
    The calls should help you get information required to complete the text.
    You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed.
    Here are some examples of API calls:
    Input: The number in the next term is 18 + 12 x 3 = 54.
    Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.
    Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
    Output: The population is 658,893 people. This is 11.4% of the national average of [Calculator(658,893 / 11.4%)] 5,763,868 people.
    Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
    Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.
    Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
    Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)] 17 years.
    Input: From this, we have 4 * 30 minutes = 120 minutes.
    Output: From this, we have 4 * 30 minutes = [Calculator(4 * 30)] 120 minutes.
    Input: [PAD]
    Output: """

    calculator_data = [
        "The population of Spain is 47.4 million, of which 7.5%, which is 3.5 million people, sleep siesta after lunch.",
        "I was born on the 24/02/2001, so today 24/02/2021 I am 20 years old.",
        "At the 2007 Math Olympics, Team Canada won 17 out of a possible 100 medals, which is 17.0% of the total medals."
    ]

    """
    PaLM(
        dim=512,
        depth=4,
        heads=8,
        dim_head=64
    )"""

    # toolformer

    toolformer = Toolformer(
        model=model,
        pad_token = tokenizer.pad_token,
        max_API_arg_length=5,
        raw_tool_prompt=prompt,
        tool_id='Calendar',
        tool=Calendar,
        tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode
    ).cuda()
    print("hey")


    # Read data from csv files good_data/calculator/0.csv, good_data/calculator/1.csv, good_data/calculator/2.csv, ...
    # Store csv files in Dataset object:
    dataset = load_dataset("/vol/bitbucket/jg2619/data/low_perplexity/", cache_dir=cache_dir, split="train")
    #dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=20)

    # invoking this will
    # (1) prompt the model with your inputs (data), inserted into [PAD] tag
    # (2) with the sampled outputs, filter out the ones that made proper API calls
    # (3) execute the API calls with the `tool` given
    # (4) filter with the specialized filter function (which can be used independently as shown in the next section)
    # (5) fine-tune on the filtered results

    print("LOADED MODEL AND DATASET")
    file_counter = 0
    
    calc_old_field_names = ['url', 'text', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket']
    calc_new_field_names = ['url', 'text', 'API_calls_text', 'API_call_response_text', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket']
    calend_old_field_names = ['url', 'text', 'date', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket'] # in database, but will add 'API_calls_text', 'API_call_response_text', 'loss_improvement', 
    calend_new_field_names = ['url', 'text', 'API_calls_text', 'API_call_response_text', 'loss_improvement', 'date', 'title', 'date_download', 'digest', 'length', 'nlines', 'source_domain', 'cc_segment', 'original_nlines', 'original_length', 'language', 'language_score', 'perplexity', 'bucket'] # in database, but will add 
    max_file_size = 5000000
    
    def file_name(tool, id):
        return f"augmented2/{tool}/{id}.csv"
    
    def data_row(processed_data, raw_data, index):
        new_row = {}
        for key in calend_old_field_names:
            new_row[key] = raw_data[key][index]
        new_row['API_calls_text'] = tokenizer.decode(processed_data['call'])
        new_row['API_call_response_text'] = tokenizer.decode(processed_data['response'])
        new_row['loss_improvement'] = processed_data['loss']
        return new_row

    # check if the file calculator_0.csv exists and create them if not with field_names as the header
    if not os.path.exists(file_name("calendar", file_counter)):
        with open(file_name("calendar", file_counter) , 'w') as f:
            writer = DictWriter(f, fieldnames=calend_new_field_names)
            writer.writeheader()
    
    generated_samples = 0
    i = 0
    
    for data in dataloader:
        print(f"Processing batch {i}")
        i += 1
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"Total memory: {t}, reserved memory: {r}, allocated memory: {a}")
        filtered_stats = toolformer(data['text'], data['date'])

        print(f"Gnerated {len(filtered_stats)} samples!!!!")
        for row in filtered_stats[:5]:
            print(tokenizer.decode(row['response']))
            print(row['loss'])
            print()
        augmented_rows = []
        for row in filtered_stats:
            augmented_rows.append(data_row(row, data, row['index']))

        if len(filtered_stats) > 0:
            with open(file_name("calendar", file_counter), 'a') as f:
                dictwriter_object = DictWriter(f, fieldnames=calend_new_field_names)

                for row in augmented_rows:
                    # Pass the dictionary as an argument to the Writerow()
                    dictwriter_object.writerow(row)
                
                generated_samples += len(filtered_stats)
                new_calendar_data = list()
            if os.path.getsize(file_name("calendar", file_counter)) > max_file_size: #
                file_counter += 1

                if not os.path.exists(file_name("calendar", file_counter)):
                    with open(file_name("calendar", file_counter), 'w') as f:
                        writer = DictWriter(f, fieldnames=calend_new_field_names)
                        writer.writeheader()

        # Empty cuda cache
        torch.cuda.empty_cache()


    raise Exception("STOP")
    
    filtered_stats = toolformer(data)
    print("Made it?????????????")
    print(filtered_stats)
    for row in filtered_stats:
        print(tokenizer.decode(row['response']))
        print(f"loss improvement: {row['loss']}")

    print("TRYING CALCULATOR")
    toolformer.update('Calculator', Calculator, calculator_prompt, 30)
    # then, once you see the 'finetune complete' message

    try:
        filtered_stats_calc = toolformer(calculator_data)
    except AssertionError as e:
        print(e)

    #response = toolformer.sample_model_with_api_calls(
    #    "How many days until the next new years?")

    # hopefully you see it invoke the calendar and utilize the response of the api call...


    wikipedia_search_prompt = f"""Your task is to complete a given piece of text.
    You can use a Wikipedia Search API to look up information.
    You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up.
    Here are some examples of API calls:
    Input: The colors on the flag of Ghana have the following meanings: red is for the blood of martyrs, green for forests, and gold for mineral wealth.
    Output: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.
    Input: But what are the risks during production of nanomaterials? Some nanomaterials may give rise to various kinds of lung damage.
    Output: But what are the risks during production of nanomaterials? [WikiSearch("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.
    Input: Metformin is the first-line drug for patients with type 2 diabetes and obesity.
    Output: Metformin is the first-line drug for [WikiSearch("Metformin first-line drug")] patients with type 2 diabetes and obesity.
    Input: [PAD]
    Output: """

    wiki_data = ["The theory of relativity, formulated by Albert Einstein, revolutionized our understanding of space, time, and gravity.",
    "The human respiratory system consists of several organs, including the lungs, bronchi, and trachea, that work together to facilitate breathing.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The periodic table of elements organizes all known chemical elements based on their atomic number and chemical properties.",
    "The concept of natural selection, proposed by Charles Darwin, explains how species evolve and adapt to their environment over time."]

    class ColBERTv2:
        def __init__(self, url: str):
            self.url = url
        def __call__(self, query, k=10):
            topk = colbertv2_get_request(self.url, query, k)
            topk = [doc['text'] for doc in topk]
            return topk

    def colbertv2_get_request(url: str, query: str, k: int):
        payload = {'query': query, 'k': k}
        res = requests.get(url, params=payload)
        topk = res.json()['topk'][:k]
        return topk

    def WikiSearch(
        input_query: str,
        url: str = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search',
        k: int = 10
    ):
        retrieval_model = ColBERTv2(url)
        output = retrieval_model(input_query, k)
        return output

        

    toolformer.update('WikiSearch', WikiSearch, wikipedia_search_prompt, 500)
    # then, once you see the 'finetune complete' message

    try:
        filtered_stats_wiki = toolformer(wiki_data)
    except AssertionError as e:
        print(e)

    response = toolformer.sample_model_with_api_calls(
        "How many days until the next new years?")