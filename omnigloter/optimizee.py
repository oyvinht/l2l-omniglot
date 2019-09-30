from __future__ import (print_function,
                        unicode_literals,
                        division)
import numpy as np
from scipy.special import comb
from pprint import pprint
from multiprocessing import Process, Queue
import time
import sys
import os
import logging
from omnigloter import config, utils, analyse_network as analysis
from omnigloter.snn_decoder import Decoder

logger = logging.getLogger("optimizee.mushroom_body")

# L2L imports
from l2l.optimizees.functions.optimizee import Optimizee
from six import iterkeys, iteritems


class OmniglotOptimizee(Optimizee):
    def __init__(self, traj, seed):

        super().__init__(traj)
        self.seed = np.uint32(seed)
        self.rng = np.random.RandomState(seed=self.seed)

        self.ind_param_ranges = config.ATTR_RANGES.copy()
        self.sim_params = traj.simulation.params.copy()

        print("In OmniglotOptimizee:")
        print(self.ind_param_ranges)
        print(self.sim_params)

    def create_individual(self):
        ipr = self.ind_param_ranges
        return {k: utils.randnum(ipr[k][0], ipr[k][1], rng=self.rng) for k in ipr}

    def bounding_func(self, individual):
        ipr = self.ind_param_ranges
        for k in ipr:
            range = ipr[k]
            val = individual[k]
            individual[k] = utils.bound(val, range)

        return individual

    def simulate(self, traj):
        work_path = traj._parameters.JUBE_params.params['work_path']
        results_path = os.path.join(work_path, 'run_results')
        os.makedirs(results_path, exist_ok=True)

        bench_start_t = time.time()
        ipr = self.ind_param_ranges
        for k in traj.par:
            try:
                print("{}:".format(k))
                print("\t{}\n".format(traj.par[k]))
            except:
                print("\tNOT FOUND!\n")

        n_out = self.sim_params['output_size']
        n_test = self.sim_params['test_per_class']
        n_class = self.sim_params['num_classes']
        n_dots = comb(n_class, 2)

        # ind_idx = np.random.randint(0, 1000)
        ind_idx = traj.individual.ind_idx
        generation = traj.individual.generation
        name = 'gen{}_ind{}'.format(generation, ind_idx)
        ind_params = {k: getattr(traj.individual, k) for k in ipr}
        print("ind_params:")
        print(ind_params)
        params = {
            'ind': ind_params,
            'sim': self.sim_params,
        }
        snn = Decoder(name, params)
        # try:
        data = snn.run_pynn()
        # except:
        #     print("error in sim, fitness = ", n_dots)
        #     return [1.0*n_dots]

        ### Analyze results
        dt = self.sim_params['sample_dt']
        out_spikes = data['recs']['output'][0]['spikes']
        labels = data['input']['labels']
        end_t = self.sim_params['duration']
        start_t = end_t - n_class * n_test * dt
        apc, ipc = analysis.spiking_per_class(labels, out_spikes, start_t, end_t, dt)

        diff_class_vectors = [np.zeros(n_out) for _ in apc]
        for c in apc:
            if len(apc[c]):
                kv = np.array(list(apc[c].keys()), dtype='int')
                diff_class_vectors[c - 1][kv] = 1


        diff_class_norms = np.linalg.norm(diff_class_vectors, axis=1)
        print("{}\tdiff vectors - norms".format(name))
        print(diff_class_norms)

        diff_class_dots = np.array([np.dot(x, y) / (diff_class_norms[ix] * diff_class_norms[iy]) \
                                    for ix, x in enumerate(diff_class_vectors) \
                                    for iy, y in enumerate(diff_class_vectors) if ix > iy])

        print("{}\tdiff dots".format(name))
        print(diff_class_dots)

        same_class_vectors = {c-1: [np.zeros(n_out) for _ in ipc[c]] for c in ipc}
        for c in ipc:
            for i, x in enumerate(ipc[c]):
                for nid in ipc[c][x]:
                    same_class_vectors[c - 1][i][nid] = 1


        same_class_norms = {c: np.linalg.norm(same_class_vectors[c], axis=1) \
                                                    for c in same_class_vectors}

        print("{}\tsame vectors - norms".format(name))
        print(same_class_norms)

        same_class_dots = {c: np.array([np.dot(x, y) / (same_class_norms[c][ix] * same_class_norms[c][iy]) \
                                for ix, x in enumerate(same_class_vectors[c]) \
                                    for iy, y in enumerate(same_class_vectors[c]) if ix > iy]) \
                                        for c in same_class_vectors}

        print("{}\tsame dots".format(name))
        print(same_class_dots)


        print("\n\nExperiment took {} seconds\n".format(time.time() - bench_start_t))

        if len(diff_class_dots) == 0:
            print("dots == 0, fitness = ", 0)

            diff_class_norms = []
            diff_class_dots = []

            same_class_norms = []
            same_class_dots = []

            diff_class_fitness = 0#n_dots
            same_class_fitness = 0

        else:
            if np.any(diff_class_norms == 0.):
                print("At least one of the norms was 0")
                whr = np.where(np.isnan(diff_class_dots))[0]
                if len(whr):
                    diff_class_dots[whr] = 1.0

                whr = np.where(np.isinf(diff_class_dots))[0]
                if len(whr):
                    diff_class_dots[whr] = 1.0

            diff_class_fitness = n_dots - np.sum(diff_class_dots)
            print("diff_fitness %s - %s = %s"%(n_dots, np.sum(diff_class_dots), diff_class_fitness))

            same_fitnesses = np.asarray([
                np.sum(same_class_dots[c]) if len(same_class_dots[c]) else 0.0 \
                                                for c in sorted(same_class_dots.keys())
            ])

            same_fitnesses[np.where(np.isnan(same_fitnesses))] = 0.0
            same_fitnesses[np.where(np.isinf(same_fitnesses))] = 0.0
            same_class_fitness = np.sum(same_fitnesses)

            print("same fitness ", same_class_fitness)

        data['analysis'] = {
            'aggregate_per_class': {
                'spikes': apc,
                'vectors': diff_class_vectors,
                'norms': diff_class_norms,
                'dots': diff_class_dots,
                'fitness': diff_class_fitness,
            },
            'individual_per_class': {
                'spikes': ipc,
                'vectors': same_class_vectors,
                'norms': same_class_norms,
                'dots': same_class_dots,
                'fitness': same_class_fitness,
            }
        }
        ### Save results for this individual

        fname = 'data_{}.npz'.format(name)
        np.savez_compressed(os.path.join(results_path, fname), **data)


        return [diff_class_fitness, same_class_fitness]
