# script that reads json files in a directory just to check if they're all fine

import os
import json

# Specify the directory
directory = '/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09_copy/'

# Iterate over the JSON files in the input directory
files = os.listdir(directory)
files.sort()
for filename in files:
    if filename.endswith('.json'):
        json_path = os.path.join(directory, filename)

        # Open the JSON file and load the data
        with open(json_path) as json_file:
            for line in json_file:
                data = json.loads(line)

        print(f'Loaded {filename}')