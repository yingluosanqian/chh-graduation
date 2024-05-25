import copy
import os
import numpy as np

from Source.Data.BoostPassSet import BoostPassSet
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Data.Benchmark import Benchmark
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans, EvalCoreset
from Source.Evaluation.model_train_eval import ModelTrainEval
from Source.OtherWork.chh.GreedyGenCoreset import GreedyGenCoreset
from Source.OtherWork.chh.InsertGreedy import InsertGreedy
from Source.Util.parallel import start_tasks
from Source.Util.util import unique, str_to_int_list, read_coreset_to_file


class ChhCoresetGen:
    def __init__(self, benchmark: Benchmark, sequence_gen: InsertGreedy, coreset_gen: GreedyGenCoreset, bpc_set):
        self.benchmark = benchmark
        self.sequence_gen = sequence_gen
        self.coreset_gen = coreset_gen
        self.coreset: Coreset = None
        self.bpc_set = bpc_set

    def run(self):
        self.sequence_gen(benchmark=self.benchmark, bpc_set=self.bpc_set, table_name=f'bpc_{self.benchmark.name}')
        self.coreset = self.coreset_gen(benchmark=self.benchmark, k=50, table_name=f'bpc_{self.benchmark.name}')


class ChhTrainEval:
    def __init__(self, train_benchmark: Benchmark, val_benchmark: Benchmark, test_benchmark: Benchmark, coreset: Coreset):
        self.train_benchmark = train_benchmark
        self.val_benchmark = val_benchmark
        self.test_benchmark = test_benchmark
        self.coreset = coreset

    def run(self, feature_type='Autophase'):
        model_train_eval = ModelTrainEval(self.train_benchmark,
                                          self.val_benchmark,
                                          self.test_benchmark,
                                          self.coreset)
        model_path = model_train_eval.train(feature_type=feature_type)
        model_train_eval.eval(model_path, feature_type=feature_type)


def main():
    # # insert greedy method
    # chh_coreset_gen = ChhCoresetGen(
    #     Benchmark('train_500', 'train_500'),
    #     InsertGreedy(),
    #     GreedyGenCoreset(),
    #     BoostPassSet(read_coreset_to_file('bpc_set.txt')),
    # )
    # chh_coreset_gen.run()
    #
    # chh_train_eval = ChhTrainEval(
    #     Benchmark('nvp_train', 'nvp_train'),
    #     Benchmark('nvp_val', 'nvp_val'),
    #     Benchmark('nvp_test', 'nvp_test'),
    #     chh_coreset_gen.coreset,
    # )
    # chh_train_eval.run('InstCountNorm')

    # rs coreset
    chh_train_eval = ChhTrainEval(
        Benchmark('nvp_train', 'nvp_train'),
        Benchmark('nvp_val', 'nvp_val'),
        Benchmark('nvp_test', 'nvp_test'),
        Coreset('rs_coreset', None),
    )
    chh_train_eval.run('InstCountNorm')

    # rs coreset
    chh_train_eval = ChhTrainEval(
        Benchmark('nvp_train', 'nvp_train'),
        Benchmark('nvp_val', 'nvp_val'),
        Benchmark('nvp_test', 'nvp_test'),
        Coreset('rs_coreset_1', None),
    )
    chh_train_eval.run('InstCountNorm')


if __name__ == '__main__':
    main()

