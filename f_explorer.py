from pypet import Environment, cartesian_product

import random
from deap import base
from deap import creator
from deap import tools
from pprint import pprint
from omnigloter.config import *

results_path = os.path.join(os.getcwd(), 'run_results')

def spiking_per_class(indices, spikes, start_t, end_t, dt):
    neurons_per_class = {u: {} for u in np.unique(indices)}
    et = 0
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        class_idx = int((st-start_t)//dt)
        cls = int(indices[class_idx])
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            if len(whr):
                narray = neurons_per_class[cls].get(nid, None)
                if narray is None:
                    narray = times[whr]
                else:
                    narray = np.append(narray, times[whr])

                neurons_per_class[cls][nid] = narray

    return neurons_per_class

def eval_one_min(trajectory):
    individual = trajectory.parameters.ind_idx
    generation = trajectory.parameters.generation
    name = 'gen{}_ind{}'.format(generation, individual)
    ind_params = {attr: trajectory.individual[i] for attr, i in ATTR2IDX.items()}

    sim_params = {k: trajectory.parameters.simulation._leaves[k]._data
                        for k in trajectory.parameters.simulation._leaves}

    for k in ind_params:
        if k in ATTR_RANGES:
            ind_params[k]

    net_params = {
        'sim': sim_params,
        'ind': ind_params
    }

    # print("\n\n%s"%(name))
    pprint(net_params)

    ex = Executor()
    data = ex.run(name, net_params)

    # print(data)

    os.makedirs(results_path, exist_ok=True)
    fname = 'data_{}.npz'.format(name)
    np.savez_compressed(os.path.join(results_path, fname), **data)

    n_out = sim_params['output_size']
    n_test = sim_params['test_per_class']
    n_class = sim_params['num_classes']
    dt = sim_params['sample_dt']
    out_spikes = data['recs']['output'][0]['spikes']
    labels = data['input']['labels']
    end_t = sim_params['duration']
    start_t = end_t - n_class * n_test * dt
    npc = spiking_per_class(labels, out_spikes, start_t, end_t, dt)
    vectors = [np.zeros(n_out) for _ in npc]
    for c in npc:
        if len(npc[c]):
            kv = np.array(list(npc[c].keys()), dtype='int')
            vectors[c - 1][kv] = 1

    norms = np.linalg.norm(vectors, axis=1)
    if np.sum(norms) == 0.:
        return [INF]

    dots = [np.dot(x, y) / (norms[ix] * norms[iy])\
                for ix, x in enumerate(vectors) \
                for iy, y in enumerate(vectors) if ix > iy]

    if len(dots) == 0:
        return [INF]

    return [np.sum(dots)]




######################################################################
######################################################################
######################################################################

def main():
    ### setup an experimental environment
    env = Environment(trajectory='FocalExplorer',
                      comment='Experiment to see which is the minimum weight'
                            'is required by a neuron to spike',
                      add_time=False, # We don't want to add the current time to the name,
                      # log_config='DEFAULT',
                      log_level=50,  # only display ERRORS
                      multiproc=multiproc,
                      ncores=n_procs, # Author's laptop had 2 cores XP
                      filename='./hdf5/', # We only pass a folder here, so the name is chosen
                      overwrite_file=True,
                      ### from the Brian2 example
                      continuable=False,
                      lazy_debug=False,
                      # use_pool=False, # We cannot use a pool, our network cannot be pickled
                      # wrap_mode='QUEUE',
                      
                      ### from DEAP example
                      automatic_storing=False,  # This is important, we want to run several
                      # batches with the Environment so we want to avoid re-storing all
                      # data over and over again to save some overhead.
                      )

    ### Get the trajectory object for the recently created envirnoment
    traj = env.trajectory

    ### genetic algorithm parameters
    traj.f_add_parameter('popsize', 10, comment='Population size')
    traj.f_add_parameter('CXPB', 0.5, comment='Crossover term')
    traj.f_add_parameter('MUTPB', 0.2, comment='Mutation probability')
    traj.f_add_parameter('NGEN', 10, comment='Number of generations')

    traj.f_add_parameter('generation', 0, comment='Current generation')
    traj.f_add_parameter('ind_idx', 0, comment='Index of individual')
    
    ### in our case we only optimize a single weight, so length == 1
    ### we need at least 2?
    traj.f_add_parameter('ind_len', 1, comment='Length of individual')

    traj.f_add_parameter('indpb', 0.05, comment='Mutation parameter')
    traj.f_add_parameter('tournsize', 3, comment='Selection parameter')

    traj.f_add_parameter('seed', 42, comment='Seed for RNG')
    
    traj.f_add_parameter('simulation.duration', N_CLASSES*TOTAL_SAMPLES*SAMPLE_DT)#ms
    traj.f_add_parameter('simulation.sample_dt', SAMPLE_DT)#ms
    traj.f_add_parameter('simulation.input_shape', INPUT_SHAPE)#rows, cols
    traj.f_add_parameter('simulation.input_divs', INPUT_DIVS)#rows, cols
    traj.f_add_parameter('simulation.input_layers', N_INPUT_LAYERS)
    traj.f_add_parameter('simulation.num_classes', N_CLASSES)
    traj.f_add_parameter('simulation.samples_per_class', N_SAMPLES)
    traj.f_add_parameter('simulation.test_per_class', N_TEST)
    traj.f_add_parameter('simulation.num_epochs', N_EPOCHS)
    traj.f_add_parameter('simulation.total_per_class', TOTAL_SAMPLES)
    traj.f_add_parameter('simulation.kernel_width', KERNEL_W)
    traj.f_add_parameter('simulation.kernel_pad', PAD)
    traj.f_add_parameter('simulation.output_size', OUTPUT_SIZE)
    traj.f_add_parameter('simulation.use_gabor', USE_GABOR_LAYER)
    traj.f_add_parameter('simulation.conn_dist', CONN_DIST)
    traj.f_add_parameter('simulation.prob_noise', PROB_NOISE_SAMPLE)


    traj.f_add_parameter('simulation.spikes_path',
        # '/home/gp283/brainscales-recognition/codebase/images_to_spikes/omniglot/spikes'
        # '/home/gp283/brainscales-recognition/codebase/images_to_spikes/omniglot/spikes_shrink'
        '/home/gp283/brainscales-recognition/codebase/images_to_spikes/omniglot/spikes_shrink_%d'%INPUT_SHAPE[0]
        # '/home/gp283/brainscales-recognition/codebase/images_to_spikes/mnist-db/spikes'
        )
    traj.f_add_parameter('simulation.database',
                         'Alphabet_of_the_Magi'
                         # 'train'
                         )
    # Placeholders for individuals and results that are about to be explored
    traj.f_add_derived_parameter('individual', [0 for x in range(traj.ind_len)],
                                 'An indivudal of the population')
    traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')

    ### setup DEAP minimization
    ### Name of our fitness function, base class from which to inherit,
    ### weights are the importance of each element in the fitness function
    ### return values; -1.0 because it's a minimization problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    
    ### Name of our individual object, it inherits from a list and has a
    ### fitness attribute which points to the recently created fitness func
    creator.create("Individual", list, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    # Attribute generator

    attr_list = [None for _ in ATTR2IDX]
    for attr in ATTR2IDX:
        r = ATTR_RANGES[attr]
        f = np.random.uniform if np.issubdtype(type(r[0]), np.floating) else randint_float
        if len(r) == 2:
            toolbox.register(attr, f, r[0], r[1])
        elif len(r) == 3:
            toolbox.register(attr, f, r[0], r[1], r[2])

        attr_list[ATTR2IDX[attr]] = getattr(toolbox, attr)

    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     attr_list, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.01, indpb=traj.indpb)
    toolbox.register("select", tools.selTournament, tournsize=traj.tournsize)
    toolbox.register("evaluate", eval_one_min)
    toolbox.register("map", env.run)  # We pass the individual as part of traj, so


    # ------- Initialize Population -------- #
    random.seed(traj.seed)

    pop = toolbox.population(n=traj.popsize)
    CXPB, MUTPB, NGEN = traj.CXPB, traj.MUTPB, traj.NGEN
    
    print("Start of evolution")
    for g in range(traj.NGEN):

        # ------- Evaluate current generation -------- #
        print("-- Generation %i --" % g)

        # Determine individuals that need to be evaluated
        eval_pop = [ind for ind in pop if not ind.fitness.valid]

        # Add as many explored runs as individuals that need to be evaluated.
        # Furthermore, add the individuals as explored parameters.
        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`:
        # This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        # prod = cartesian_product({'generation': [g],
        #                           'ind_idx': range(len(eval_pop))})
        prod = cartesian_product({'generation': [g],
                                  'ind_idx': range(len(eval_pop)),
                                  'individual':[list(x) for x in eval_pop]},
                                  [('ind_idx', 'individual'),'generation'])
        # pprint(prod)
        traj.f_expand(prod)

        fitnesses_results = toolbox.map(toolbox.evaluate)  # evaluate using our fitness function

        # fitnesses_results is a list of
        # a nested tuple: [(run_idx, (fitness,)), ...]
        for idx, result in enumerate(fitnesses_results):
            # Update fitnesses
            _, fitness = result  # The environment returns tuples: [(run_idx, run), ...]
            eval_pop[idx].fitness.values = fitness

        # Append all fitnesses (note that DEAP fitnesses are tuples of length 1
        # but we are only interested in the value)
        pprint([x.fitness.values for x in eval_pop])

        # Gather all the fitnesses in one list and print the stats
        fits = [x.fitness.values[0] if len(x.fitness.values) else 0 for x in eval_pop]
        traj.fitnesses.extend(fits)

        print("  Evaluated %i individuals" % len(fitnesses_results))

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


        # ------- Create the next generation by crossover and mutation -------- #
        if g < traj.NGEN -1:  # not necessary for the last generation
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # The population is entirely replaced by the offspring
            pop[:] = offspring

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    traj.f_store()  # We switched off automatic storing, so we need to store manually
    
    

if __name__  == '__main__':
    main()


