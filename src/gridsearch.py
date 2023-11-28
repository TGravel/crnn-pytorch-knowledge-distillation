from distill import main as distill
import os
import itertools
import json
import datetime
from config import gridsearch_config as config

def main():
    result = []
    keys, values = zip(*config.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for i in permutations_dicts:
        result.append(distill(i))
    print(result)
    #filename = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + ".json"
    #with open(os.path.join('gridresults', filename), 'wb+') as f:
    #    json.dump(result, f)


if __name__ == '__main__':
    main()
