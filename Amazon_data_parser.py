import json
import gzip
import re

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

for l in parse(R"F:\Dataset\Amazon\Electonics\meta_Electronics.json.gz"):
    print(l)
    data = (json.loads(l))
    print(data['categories'])
    print('\n====================================')
    stop = 1
    pass

    # https://www.amazon.in/dp/0511189877