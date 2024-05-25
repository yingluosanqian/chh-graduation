import copy

import numpy as np

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans
from Source.Util.parallel import start_tasks
from pathlib import Path

from Source.Util.util import read_coreset_to_file, save_coreset_to_file


class GreedyGenCoreset:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def get_matrix_row(self, pss, seqs, sql_list):
        env = EnvManager()
        res1 = [env.evaluate_for_mp(Trans.copy_and_reset(pss, seq), sql_list) for seq in seqs]
        res2 = env.evaluate_for_mp(Trans.copy_and_reset(pss, []), sql_list)
        res = [e / res2 if res2 != 0 else 1.0 for e in res1]
        return res

    def get_matrix(self, benchmark: Benchmark):
        seqs = [pss.sequence for pss in benchmark.pss_list]
        args = [(pss, seqs) for pss in benchmark.pss_list]
        res = start_tasks(self.get_matrix_row, args, desc='Matrix')
        return np.array(res).T

    def solve(self, benchmark: Benchmark, k=50, table_name=None):
        data_path = Path('.') / 'data'
        data_path.mkdir(exist_ok=True)
        coreset_name = f'insert_greedy_{table_name}'
        coreset_path = data_path / coreset_name
        if not coreset_path.is_file():
            seqs = [pss.sequence for pss in benchmark.pss_list]
            matrix = self.get_matrix(benchmark)
            coreset_index = []
            for it in range(k):
                mx, val = 0, -float("inf")
                for idx, seq in enumerate(seqs):
                    if idx in coreset_index:
                        continue
                    cp_coreset_index = copy.deepcopy(coreset_index)
                    cp_coreset_index.append(idx)
                    perf_sum = np.sum(np.max(matrix[cp_coreset_index], axis=0))
                    if perf_sum > val:
                        mx, val = idx, perf_sum
                coreset_index.append(mx)
                # print(f'{it}-th Iteration..., {mx + 1}-th sequence was chosen.')

            coreset_seq = [seqs[seq_index] for seq_index in coreset_index]
            save_coreset_to_file(coreset_seq, coreset_path)
        return Coreset(name=coreset_name, seq_list=read_coreset_to_file(coreset_path))
