import copy
from Source.Data.Benchmark import Benchmark
from Source.Data.BoostPassSet import BoostPassSet
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans
from Source.Util.parallel import start_tasks
from Source.Util.util import GLOBAL_VALUE


class InsertGreedy:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def select_main(self, pro_seq_siz: ProSeqSiz, bpc_set, env, sql_list):
        """
        @Todo: implement your greedy algorithm, choose a pass and a pos, and insert pass to seq[pos]
        :return just return the ProSeqSiz, my english is poor
        """
        pss_list = []
        for i in range(len(pro_seq_siz.sequence) + 1):
            for main in bpc_set:
                new_seq = copy.deepcopy(pro_seq_siz.sequence)
                new_seq.insert(i, main)
                pss_list.append(Trans.copy_and_reset(pro_seq_siz, new_seq))
        size_list = env.multi_evaluate_for_mp(pss_list, sql_list)
        result_pss_list = [ProSeqSiz.import_pss(pss, size) for pss, size in zip(pss_list, size_list)]
        result = min(result_pss_list)

        return result

    def solve_one(self, pss: ProSeqSiz, bpc_set, sql_list):
        """
        @Todo: implement your greedy algorithm, the following code provide an example of naive greedy algorithm
        :return: a ProSeqSiz with optimal pass sequence
        """
        env = EnvManager()
        optimal_pss = copy.deepcopy(pss)
        optimal_pss.sequence = []

        step = 8  # we set max step as 16
        for _ in range(step):
            new_res = self.select_main(optimal_pss, bpc_set, env, sql_list)

            if new_res < optimal_pss:
                optimal_pss = new_res
            else:
                break

        return optimal_pss

    def solve(self, benchmark: Benchmark, bpc_set, table_name=None):
        print(f'Now, running insert greedy algorithm.')
        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            if benchmark.load_from_db(table_name):
                # benchmark.debug()  # debug
                return

        benchmark.sort_by_size()
        args = [(pss, bpc_set, ) for pss in benchmark.pss_list]
        res = start_tasks(self.solve_one, args, desc='Insert Greedy', order=False)
        benchmark.pss_list = res

        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            benchmark.save_to_db(table_name)
