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
from time import sleep
import logging
from omnigloter import config, utils, analyse_network as analysis
from omnigloter.snn_decoder import Decoder

results_path = os.path.join(os.getcwd(), 'run_results')


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

        ind_idx = np.random.randint(0, 1000)
        generation = traj.parameters.generation
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



        ### Save results for this individual
        os.makedirs(results_path, exist_ok=True)
        fname = 'data_{}.npz'.format(name)
        np.savez_compressed(os.path.join(results_path, fname), **data)


        ### Analyze results
        dt = self.sim_params['sample_dt']
        out_spikes = data['recs']['output'][0]['spikes']
        labels = data['input']['labels']
        end_t = self.sim_params['duration']
        start_t = end_t - n_class * n_test * dt
        apc, ipc = analysis.spiking_per_class(labels, out_spikes, start_t, end_t, dt)
        vectors = [np.zeros(n_out) for _ in apc]
        for c in apc:
            if len(apc[c]):
                kv = np.array(list(apc[c].keys()), dtype='int')
                vectors[c - 1][kv] = 1


        print("{}\tvectors".format(name))
        print(vectors)


        norms = np.linalg.norm(vectors, axis=1)
        print("{}\tvectors - norms".format(name))
        print(norms)

        dots = np.array([np.dot(x, y) / (norms[ix] * norms[iy])\
                            for ix, x in enumerate(vectors) \
                            for iy, y in enumerate(vectors) if ix > iy])

        print("{}\tdots".format(name))
        print(dots)

        print("\n\nExperiment took {} seconds\n".format(time.time() - bench_start_t))

        if len(dots) == 0:
            print("dots == 0, fitness = ", n_dots)
            return [1.0*n_dots]

        if np.any(norms == 0.):
            print("At least one of the norms was 0")
            whr = np.where(np.isnan(dots))[0]
            if len(whr):
                dots[whr] = 1.0

            whr = np.where(np.isinf(dots))[0]
            if len(whr):
                dots[whr] = 1.0


        fitness = np.sum(dots)
        print("fitness ", fitness)
        return [fitness]