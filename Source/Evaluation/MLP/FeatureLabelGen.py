import copy

import numpy as np

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans

from Source.Util.parallel import start_tasks
from Source.Util.util import get_root_path


class FLGen:

    def __init__(self, benchmark: Benchmark, coreset: Coreset, name):
        self.benchmark = benchmark
        self.coreset = coreset
        self.name = name
        self.bad_benchmark_path = f'{get_root_path()}/Train/BadBenchmark/temp_bad'
        with open(self.bad_benchmark_path, 'w') as _:
            pass

    def gen_feature(self):
        def gen_one(pss: ProSeqSiz, sql_list):
            try:
                env = EnvManager()
                init = env.evaluate_for_mp(Trans.copy_and_reset(pss, []), sql_list)
                ft = []
                for i in range(EnvManager.n):
                    ft.append((init - env.evaluate_for_mp(Trans.copy_and_reset(pss, [i]), sql_list)) / init)
                return ft
            except Exception as _:
                return str(pss.name.split()[1])
            finally:
                pass

        args_list = [(pss,) for pss in self.benchmark.pss_list]
        temp_res = start_tasks(gen_one, args_list, desc='Run each pass')
        res = []
        for e in temp_res:
            if isinstance(e, str):
                with open(self.bad_benchmark_path, 'a') as f:
                    f.write(f'{e}\n')
            else:
                if len(e) != 124:
                    print(e)
                res.append(e)
        res = np.array(res)
        np.savetxt(f"{get_root_path()}/Evaluation/MLP/data_124/{self.name}_X.csv", res, delimiter=",")

    def gen_label(self):
        def solve_one(pss, all_seq, sql_list):
            env = EnvManager()
            try:
                init = env.evaluate_for_mp(Trans.copy_and_reset(pss, []), sql_list)
                size_list = []
                for seq in all_seq:
                    tmp_pss = Trans.copy_and_reset(pss, seq)
                    size = env.evaluate_for_mp(tmp_pss, sql_list)
                    size_list.append(init - size)

                return size_list
            except Exception as _:
                return pss[0].split()[1]
            finally:
                pass

        args_list = [(pss, self.coreset) for pss in self.benchmark.pss_list]
        temp_res = start_tasks(solve_one, args_list, desc='Evaluation')
        res = []
        for e in temp_res:
            if isinstance(e, str):
                with open(f'{get_root_path()}/Train/BadBenchmark/temp_bad', 'a') as f:
                    f.write(f'{e}\n')
            else:
                res.append(e)
        res = np.array(res)
        np.savetxt(f"{get_root_path()}/Evaluation/MLP/data_124/{self.name}_Y.csv", res, delimiter=",")


if __name__ == '__main__':
    # start_ray()

    # for pass_set, pass_set_name in [
    #     (f'final.txt', f'final'),
    #     (f'coreset.txt', f'coreset'),
    # ]:
    #     flg = FLGen(
    #         Benchmark('__train', '__train'),
    #         Coreset(pass_set),
    #         f'train_{pass_set_name}',
    #     )
    #     flg.gen_feature()
    #     flg.gen_label()
    #
    #     flg = FLGen(
    #         Benchmark('__val', '__val'),
    #         Coreset(pass_set),
    #         f'val_{pass_set_name}',
    #     )
    #     flg.gen_feature()
    #     flg.gen_label()

    # test
    flg = FLGen(
        Benchmark('__test', '__test'),
        None,
        f'test',
    )
    flg.gen_feature()
