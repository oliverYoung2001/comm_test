import json
import re

GPU_DIST = '412'
BACKEND = 'NCCL'

def log2json2sorted_all():
    # file_name = f'single_turn_{GPU_DIST}_{BACKEND}'
    file_in = f'./results/single_turn.log'
    # file_out = f'./results/{file_name}.json'
    # GPU_NUM: 4
    # PROC_NUM: 4
    # NODE_NUM: 1
    # NTASK_PER_NODE: 4
    pattern_GPU_NUM = re.compile(f'GPU_NUM: (\d+)', re.I)
    pattern_NODE_NUM = re.compile(f'NODE_NUM: (\d+)', re.I)
    pattern_NTASK_PER_NODE = re.compile(f'NTASK_PER_NODE: (\d+)', re.I)
    pattern_BACKEND = re.compile(f'BACKEND: (\S*)', re.I)
    
    pattern_head = re.compile(r'ringid_arrays.size\(\): (\d+)', re.I)
    pattern_data = re.compile(r'([\d|\|]*): (\S*) ([\d|\.]*) s, (\S*) ([\d|\.]*) GB/s, (\S*) ([\d|\.]*) KB, (\S*) ([\d|\.]*) GB/s', re.I)
    
    with open(file_in, 'r', encoding='utf-8') as f:
        i = 0
        lines = f.readlines()
        while True:
            pattern_num = 0
            data = []
            GPU_NUM = 0
            NODE_NUM = 0
            NTASK_PER_NODE = 0
            BACKEND = ''
            while pattern_num == 0:
                if i >= len(lines):
                    break
                match_obj = re.match(pattern_GPU_NUM, lines[i])
                if match_obj:
                    GPU_NUM = int(match_obj.group(1))
                match_obj = re.match(pattern_NODE_NUM, lines[i])
                if match_obj:
                    NODE_NUM = int(match_obj.group(1))
                match_obj = re.match(pattern_NTASK_PER_NODE, lines[i])
                if match_obj:
                    NTASK_PER_NODE = int(match_obj.group(1))
                match_obj = re.match(pattern_BACKEND, lines[i])
                if match_obj:
                    BACKEND = match_obj.group(1)
                match_obj = re.match(pattern_head, lines[i])
                if match_obj:
                    pattern_num = int(match_obj.group(1))
                i += 1
                
            if i >= len(lines):
                break
            assert(pattern_num > 0)
            if pattern_num == 0:
                print("Error: Not Fild pattern_num !!!")
                assert(False)
            assert(GPU_NUM > 0)
            assert(NODE_NUM > 0)
            assert(NTASK_PER_NODE > 0)
            assert(len(BACKEND) > 0)
            assert(GPU_NUM == NODE_NUM * NTASK_PER_NODE)
            NUMA_NUM = NTASK_PER_NODE // 4
            GPU_DIST = f'{NTASK_PER_NODE // NUMA_NUM}{NUMA_NUM}{NODE_NUM}'
            
            while pattern_num > 0:
                if i >= len(lines):
                    break
                match_obj = re.match(pattern_data, lines[i])
                if match_obj:
                    groups = match_obj.groups()
                    entry = {'pat': groups[0]}
                    for j in range(1, len(groups), 2):
                        entry[groups[j]] = float(groups[j + 1])
                    data.append(entry)
                    pattern_num -= 1
                i += 1
            assert(pattern_num == 0)
            data.sort(key=lambda x: x['REAL_BD'], reverse=True)
            file_out = f'./results/single_turn_{GPU_DIST}_{BACKEND}_sorted.list'
            with open(file_out, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(f"{entry['pat']}: {entry['REAL_BD']} GB/s\n")

    
def log2json(GPU_DIST, BACKEND):
    file_name = f'single_turn_{GPU_DIST}_{BACKEND}'
    file_in = f'./results/{file_name}.log'
    file_out = f'./results/{file_name}.json'
    pattern_head = re.compile(r'ringid_arrays.size\(\): (\d+)', re.I)
    pattern_data = re.compile(r'([\d|\|]*): (\S*) ([\d|\.]*) s, (\S*) ([\d|\.]*) GB/s, (\S*) ([\d|\.]*) KB, (\S*) ([\d|\.]*) GB/s', re.I)
    pattern_num = 0
    total_num = 0
    data = []
    with open(file_in, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            # if i >= 2:
            #     break
            # print(f'line: {line}')
            match_obj = re.match(pattern_data, line)
            if match_obj:
                groups = match_obj.groups()
                single_data = {'pat': groups[0]}
                for j in range(1, len(groups), 2):
                    single_data[groups[j]] = float(groups[j + 1])
                data.append(single_data)
                # print(f'match_obj.groups: {match_obj.groups()}')
                total_num += 1
            else:
                match_obj = re.match(pattern_head, line)
                if match_obj:
                    pattern_num = int(match_obj.group(1))
                # else:
                #     # print(f'line: {line}')
                #     # print(f'i: {i}')
            
    assert(len(data) == total_num)
    print(f"total_num: {total_num}, pattern_num: {pattern_num}")
    # assert(total_num == pattern_num)
    with open(file_out, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
def json2sorted(GPU_DIST, BACKEND):
    file_name = f'single_turn_{GPU_DIST}_{BACKEND}'
    file_in = f'./results/{file_name}.json'
    file_out = f'./results/{file_name}_sorted.log'
    with open(file_in, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.sort(key=lambda x: x['REAL_BD'], reverse=True)
    with open(file_out, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(data):
            f.write(f"{entry['pat']}: {entry['REAL_BD']} GB/s\n")
    
def main():
    log2json2sorted_all()
    # log2json(GPU_DIST, BACKEND)
    # json2sorted(GPU_DIST, BACKEND)
    
    
if __name__ == '__main__':
    main()