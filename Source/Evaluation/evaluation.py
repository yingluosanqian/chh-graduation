import copy

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import get_geo_mean, get_arith_mean, get_benchmark_name


class Trans:
    @staticmethod
    def copy_and_reset(pss: ProSeqSiz, seq: list[int]) -> tuple:
        pro_seq2 = copy.deepcopy(pss)
        pro_seq2.sequence = seq
        return pro_seq2.export()


class EvalCoreset:
    @staticmethod
    def solve_one(pss, coreset, sql_list):
        env = EnvManager()
        res = []
        for seq in coreset:
            new_pss = Trans.copy_and_reset(pss, seq)
            size = env.evaluate_for_mp(new_pss, sql_list)
            res.append(ProSeqSiz.import_pss(new_pss, size))
            # res.append(ProSeqSiz.import_pss(*(env.evaluate_for_mp_with_cut(new_pss, sql_list))))
        return min(res)

    @staticmethod
    def evaluate(test_benchmark: Benchmark, coreset: Coreset, filepath=None):
        args = [(pss, coreset) for pss in test_benchmark.pss_list]
        test_benchmark.pss_list = start_tasks(EvalCoreset.solve_one, args, desc='Eval Coreset')
        if filepath is None:
            filepath = f'result.txt'
        dataset = {}
        for pss in test_benchmark.pss_list:
            name = get_benchmark_name(pss.name)
            if name not in dataset:
                dataset[name] = []
            dataset[name].append(pss)

        geo_ls, arith_ls = [], []
        with open(filepath, 'w') as _:
            pass
        for dataset_name, _pss_list in dataset.items():
            benchmark = Benchmark(dataset_name, _pss_list)
            with open(filepath, 'a') as f:
                geo, arith = benchmark.get_geo_mean(), benchmark.get_arith_mean()
                f.write(f'benchmark={benchmark.name}, '
                        f'{"{:.3f}".format(geo)}/'
                        f'{"{:.1%}".format(arith)}\n')
                geo_ls.append(geo)
                arith_ls.append(arith)
        with open(filepath, 'a') as f:
            f.write(f'geo_mean={"{:.3f}".format(get_geo_mean(geo_ls))} '
                    f'arith_mean={"{:.1%}".format(get_arith_mean(arith_ls))}'
                    f'\n')
        test_benchmark.save(None, None, record=False, file_path=f'{filepath}_detail.txt')


if __name__ == '__main__':
    for set_file, set_name in [
        # ('train_a.txt', 'train_a'),
        # ('coreset_ex.txt', 'coreset'),
        # ('cf.txt', 'cf'),
        # ('final.txt', 'final'),
        # ('bpc_coreset_greedy.txt', 'bpc_coreset_greedy'),
        # ('bpc_coreset_binary.txt', 'bpc_coreset_binary'),
        ('bpc_coreset_noprobdd.txt', 'bpc_coreset_noprobdd'),
    ]:
        coreset = Coreset(f'{set_file}')
        EvalCoreset.evaluate(Benchmark('__test', '__test'), coreset, filepath=f'results_{set_name}.txt')

    pass
