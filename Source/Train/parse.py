import copy
import json
import random
from pathlib import Path

from tqdm import tqdm

from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import str_to_int_list


def calc(benchmarks):
    count = {}
    for benchmark in benchmarks:
        name = benchmark.split('//')[1].split('/')[0]
        if name not in count.keys():
            count[name] = []
        count[name].append(benchmark)
    return count


def extract_benchmark(json_file_path, save_file_path, is_test=False):
    with json_file_path.open('r') as f:
        data = json.load(f)
    if not is_test:
        with save_file_path.open('w') as f:
            for benchmark in data['benchmark_dataset']['benchmarks']:
                f.write(f'Online {benchmark}\n')
    else:
        with save_file_path.open('w') as f:
            for benchmark in data['benchmarks']:
                f.write(f'Online {benchmark}\n')


# def parse_val(name):
#     with json_file_path.open('r') as f:
#         data = json.load(f)
#     nvp_train = Path('Benchmark') / 'nvp_train.txt'
#     with nvp_train.open('w') as f:
#         for benchmark in data['benchmark_dataset']['benchmarks']:
#             f.write(f'Online {benchmark}\n')


def parse_test(name):
    print()
    print('============== Test ==============')
    print('Loading, this may cost some time...')
    with open(name, 'r') as f:
        data = json.load(f)
    print('Loading completed.')
    print('Now, saving those benchmark to local files.')
    benchmarks = data['benchmarks']
    # print(benchmarks)
    count = calc(benchmarks)
    for k, v in count.items():
        print(k, len(v))

    with open(f'Benchmark/test_size.analysis', 'w') as _:
        pass

    for k, v in tqdm(count.items()):
        tasks = [(e, e, []) for e in random.sample(v, min(len(v), 300))]
        task_res = EnvManager.multi_evaluate(tasks)
        avg = sum(task_res) / len(task_res)
        with open(f'Benchmark/test_size.analysis', 'a') as f:
            f.write(f'avg of {k} is {avg}\n')
        # with open(f'Benchmark/__test_{k}.txt', 'a') as f:
        #     for benchmark in v:
        #         f.write(f'Online {benchmark}\n')


def gen_my_test_json(name, output_name):
    print('============== Loading ==============')
    print('Loading, this may cost some time...')
    with open(name, 'r') as f:
        data = json.load(f)

    # with open(f'Benchmark/__origin_test.txt', 'w') as f:
    #     for benchmark in data['benchmarks']:
    #         f.write(f'Online {benchmark}\n')

    test_benchmarks = []
    with open(f'Benchmark/__test.txt', 'r') as f:
        for line in f:
            pos, name = line.strip().split(' ')
            if pos == 'Online':
                test_benchmarks.append(name)

    print('============== Modify ==============')
    print('Modifying, this may cost some time...')
    data['benchmarks'] = test_benchmarks

    # 将数据写入 JSON 文件
    print('Dumping, this may cost some time...')
    with open(f'{output_name}', "w") as f:
        json.dump(data, f, indent=4)

    # with open(f'Benchmark/__test_small.txt', 'w') as f:
    #     for benchmark in data['benchmarks']:
    #         f.write(f'Online {benchmark}\n')


def gen_my_json(name, output_name, k=998244353):

    def solve_one(pss: ProSeqSiz, all_seq, sql_list):
        def trans(pss_tmp: ProSeqSiz, seq_tmp: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(pss_tmp)
            pro_seq2.sequence = seq_tmp
            return pro_seq2.export()

        env = EnvManager()
        pss_list = []
        for seq in all_seq:
            tmp_pss = trans(pss, seq)
            size = env.evaluate_for_mp(tmp_pss, sql_list)
            pss_list.append(ProSeqSiz.import_pss(tmp_pss, size))

        return pss_list

    print()
    print('============== Train ==============')
    print('Loading, this may cost some time...')
    with open(name, 'r') as f:
        data = json.load(f)

    coreset = []
    with open(f'../Evaluation/coreset/final.txt', 'r') as f:
        for line in f:
            coreset.append(str_to_int_list(line))

    print('============== Modify ==============')
    print('Loading, this may cost some time...')
    data["action_sequences"]["actionseqs"] = [{"actions": seq} for seq in coreset]

    tmp = [e["benchmark"] for e in data["samples"]]
    tmp = random.sample(tmp, min(len(tmp), k))
    data['benchmark_dataset']['benchmarks'] = tmp
    args = [(ProSeqSiz(f'Online {name}', None, []), coreset) for name in tmp]
    res = {name: [pss.get_size() for pss in pss_list] for name, pss_list in zip(tmp, start_tasks(solve_one, args, desc='Parse Train'))}

    del_idx = []
    with tqdm(total=len(tmp), desc='cal size') as bar:
        for idx, e in enumerate(data["samples"]):
            benchmark = e["benchmark"]
            if benchmark in tmp:
                e["all_ir_searches"] = [int(e) for e in res[benchmark]]
                e["best_actionseq_idx"] = res[benchmark].index(min(res[benchmark]))
                bar.update(1)
            else:
                del_idx.append(idx)
    del_idx.sort(reverse=True)
    for idx in del_idx:
        del data["samples"][idx]
    # 将数据写入 JSON 文件
    print('Dumping, this may cost some time...')
    with open(f'{output_name}', "w") as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    random.seed(998244353)
    extract_benchmark(Path('CoresetNVP') / 'trajdataset_all10k-train-medium-all10k.json',
                      Path('Benchmark') / 'nvp_train.txt')
    extract_benchmark(Path('CoresetNVP') / 'trajdataset_all10k-val-medium-all10k.json',
                      Path('Benchmark') / 'nvp_val.txt')
    extract_benchmark(Path('CoresetNVP') / 'benchmarkdataset_all-test.json',
                      Path('Benchmark') / 'nvp_test.txt',
                      is_test=True)

    # gen_my_json(f'coreset_train.json', 'cyh_train.json')
    # gen_my_json(f'coreset_val.json', 'cyh_val.json')
    # gen_my_test_json(f'coreset_test.json', 'cyh_test.json')
    pass
