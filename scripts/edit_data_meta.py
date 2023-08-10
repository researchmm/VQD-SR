in_file = '/home/v-wenjiwang/zx_container/code/VQD-SR/taming-transformers/data/test.txt'
out_file = '/home/v-wenjiwang/zx_container/code/VQD-SR/taming-transformers/data/test_.txt'

with open(in_file, 'r') as f:
    lines = f.readlines()

with open(out_file, 'w') as f:
    for line in lines:
        line = line.replace('/AVC-RealLQ', 'AVC-RealLQ')
        f.write(line)