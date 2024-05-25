import copy
from pathlib import Path

import numpy
import numpy as np
import torch
from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.MLP.MLP4Autophase import model_loading, mlp_predict, MLP, MLPWithAttention
from Source.Util.parallel import start_tasks
from Source.Util.util import get_geo_mean, get_arith_mean


class ModelEvaluation:
    @staticmethod
    def load_X(X: numpy.array):
        X = torch.FloatTensor(X)
        return X

    @staticmethod
    def get_benchmark_name(name):
        if 'benchmark:' in name or 'generator:' in name:
            return name.split('/')[2]
        elif 'abc' in name and '.cpp' in name:
            return 'atcoder'
        elif '.cpp' in name:
            return 'codeforces'

    @staticmethod
    def evaluate_coreset(total_benchmark: Benchmark, X_test, coreset: Coreset, model, model_path: Path, top_list=(1, 3, 5, 50)):
        print(f'shape of X: {X_test.shape}')
        eval_path = Path('MLP') / 'eval'
        eval_path.mkdir(parents=True, exist_ok=True)
        model_name = model_path.name
        save_file_path = eval_path / f'evaluation_result_{model_name}.txt'
        with save_file_path.open('w') as _:
            pass

        def solve_one(pss, all_seq, sql_list):
            def trans(temp: ProSeqSiz, seq: list[int]) -> tuple:
                pro_seq2 = copy.deepcopy(temp)
                pro_seq2.sequence = seq
                return pro_seq2.export()

            env = EnvManager()
            pss_list = []
            for seq in all_seq:
                tmp_pss = trans(pss, seq)
                size = env.evaluate_for_mp(tmp_pss, sql_list)
                pss_list.append(ProSeqSiz.import_pss(tmp_pss, size))

            return pss_list

        args_list = [(pss, coreset) for pss in total_benchmark.pss_list]
        res = start_tasks(solve_one, args_list, desc='Evaluation')
        for top in top_list:
            if top == 0:
                continue
            dataset = {}
            predictions = mlp_predict(model, X_test, top)
            n = len(predictions)
            for i in range(n):
                indices = predictions[i].tolist()
                min_pss = min([res[i][idx] for idx in indices])
                name = ModelEvaluation.get_benchmark_name(min_pss.name)
                if name not in dataset:
                    dataset[name] = []
                dataset[name].append(min_pss)

            with save_file_path.open('a') as f:
                f.write(f'top = {top}\n')
            geo_ls, arith_ls = [], []
            for dataset_name, _pss_list in dataset.items():
                benchmark = Benchmark(dataset_name, _pss_list)
                with save_file_path.open('a') as f:
                    geo, arith = benchmark.get_geo_mean(), benchmark.get_arith_mean()
                    f.write(f'benchmark={benchmark.name}, '
                            f'{"{:.3f}".format(geo)}/'
                            f'{"{:.1%}".format(arith)}\n')
                    geo_ls.append(geo)
                    arith_ls.append(arith)
            with save_file_path.open('a') as f:
                f.write(f'geo_mean={"{:.3f}".format(get_geo_mean(geo_ls))} '
                        f'arith_mean={"{:.1%}".format(get_arith_mean(arith_ls))}'
                        f'\n')

        predictions = mlp_predict(model, X_test, 50)
        all_pss_list = []
        n = len(predictions)
        for i in range(n):
            indices = predictions[i].tolist()
            min_pss = min([res[i][idx] for idx in indices])
            all_pss_list.append(min_pss)
        all_benchmark = Benchmark('all', all_pss_list)
        save_file_path_all = Path('MLP') / 'eval' / f'evaluation_all_{model_name}.txt'
        all_benchmark.save(None, None, record=False, file_path=save_file_path_all)


if __name__ == '__main__':
    # start_ray()

    Coreset(f'bpc_coreset_binary.txt')
    Coreset(f'rs_coreset_1.txt')
    Coreset(f'rs_coreset_2.txt')

    # test_model = MLPWithAttention()
    # model_loading(test_model, 'ylsq_attn_mlp')
    # ModelEvaluation.evaluate_coreset(
    #     Benchmark('__test', '__test'),
    #     Coreset(f'final.txt'),
    #     'final_mlp',
    #     test_model,
    #     'ylsq_attn_mlp',
    # )

    # test_model = MLP()
    # model_loading(test_model, 'ylsq_mlp')
    # ModelEvaluation.evaluate_coreset(
    #     Benchmark('__test', '__test'),
    #     Coreset(f'final.txt'),
    #     'final_mlp',
    #     test_model,
    #     'ylsq_mlp',
    #     top_list=range(1, 51)
    # )
    #
    # test_model = MLP()
    # model_loading(test_model, 'coreset_mlp')
    # ModelEvaluation.evaluate_coreset(
    #     Benchmark('__test', '__test'),
    #     Coreset(f'coreset.txt'),
    #     'final_mlp',
    #     test_model,
    #     'coreset_mlp',
    #     top_list=range(1, 51)
    # )

    # test_model = MLPWithAttention()
    # model_loading(test_model, 'ylsq_attn_mlp')
    # ModelEvaluation.evaluate_coreset(
    #     Benchmark('__test', '__test'),
    #     Coreset(f'final.txt'),
    #     'final_mlp',
    #     test_model,
    #     'ylsq_attn_mlp',
    # )
    #
    # test_model = MLPWithAttention()
    # model_loading(test_model, 'coreset_attn_mlp')
    # ModelEvaluation.evaluate_coreset(
    #     Benchmark('__test', '__test'),
    #     Coreset(f'coreset.txt'),
    #     'final_mlp',
    #     test_model,
    #     'coreset_attn_mlp',
    # )

    pass
