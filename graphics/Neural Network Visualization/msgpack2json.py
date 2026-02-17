import msgpack
import json
import sys


with open(sys.argv[1], 'rb') as f:
    data = msgpack.unpackb(f.read(), strict_map_key=False, raw=False)
with open(sys.argv[2], 'w') as f:
    js = json.dumps(data, separators=(',', ':'))
    js = js.replace('},', '},\n').replace('],', '],\n')
    f.write(js)
