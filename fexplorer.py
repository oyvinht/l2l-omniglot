import logging.config
import os
import sys
sys.path.append('.')
sys.path.append("./omnigloter")
import numpy as np
from glob import glob

from l2l.utils.environment import Environment
from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
# from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer, RMSPropParameters
from l2l.optimizers.evolutionstrategies.optimizer import EvolutionStrategiesOptimizer, EvolutionStrategiesParameters
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from l2l.paths import Paths

from l2l.logging_tools import create_shared_logger_data, configure_loggers

from l2l.utils import JUBE_runner
from l2l import dict_to_list



from omnigloter.optimizee import OmniglotOptimizee
from omnigloter import config
# from pid_mb_rl.optimizer import GradientDescentMultiOptimizer, RMSPropParameters

logger = logging.getLogger("bin.l2l-omniglot")
GRADDESC, EVOSTRAT, GENALG = range(3)
OPTIMIZER = EVOSTRAT
ON_JEWELS = bool(0)

def main():
    name = "L2L-OMNIGLOT"
    root_dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    paths = Paths(name, dict(run_num="test"), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, "data.h5")

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(trajectory=name, filename=traj_file,
                      file_title="{} data".format(name),
                      comment="{} data".format(name),
                      add_time=bool(1),
                      automatic_storing=bool(1),
                      log_stdout=bool(0),  # Sends stdout to logs
                      multiprocessing=bool(1),
                      )
    create_shared_logger_data(logger_names=["bin", "optimizers"],
                              log_levels=["INFO", "INFO"],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # Name of the scheduler
    # traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")
    # Command to submit jobs to the schedulers
    traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")
    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "1")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "00:01:00")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "1")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "1")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "1")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address", "g.pineda-garcia@sussex.ac.uk")
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    # JUBE parameters for multiprocessing. Relevant even without scheduler.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "2")
    # The execution command
    if ON_JEWELS:
        # -N num nodes
        # -t exec time (mins)
        # -n num sub-procs
        traj.f_add_parameter_to_group("JUBE_params", "exec",
            "srun -t 20 -N 1 --exclusive -n 4 -c 1 --gpus-per-task 1 " + \
            " python3 " + os.path.join(paths.root_dir_path, "run_files/run_optimizee.py"))
    else:
        traj.f_add_parameter_to_group("JUBE_params", "exec", "python3 " + \
                                      os.path.join(paths.root_dir_path, "run_files/run_optimizee.py"))

    # Ready file for a generation
    traj.f_add_parameter_to_group("JUBE_params", "ready_file",
                                  os.path.join(paths.root_dir_path, "readyfiles/ready_w_"))
    # Path where the job will be executed
    traj.f_add_parameter_to_group("JUBE_params", "work_path", paths.root_dir_path)

    ### Maybe we should pass the Paths object to avoid defining paths here and there
    traj.f_add_parameter_to_group("JUBE_params", "paths_obj", paths)

    traj.f_add_parameter_group("simulation", "Contains JUBE parameters")
    traj.f_add_parameter_to_group("simulation", 'duration', config.DURATION)#ms
    traj.f_add_parameter_to_group("simulation", 'sample_dt', config.SAMPLE_DT)#ms
    traj.f_add_parameter_to_group("simulation", 'input_shape', config.INPUT_SHAPE)#rows, cols
    traj.f_add_parameter_to_group("simulation", 'input_divs', config.INPUT_DIVS)#rows, cols
    traj.f_add_parameter_to_group("simulation", 'input_layers', config.N_INPUT_LAYERS)
    traj.f_add_parameter_to_group("simulation", 'num_classes', config.N_CLASSES)
    traj.f_add_parameter_to_group("simulation", 'samples_per_class', config.N_SAMPLES)
    traj.f_add_parameter_to_group("simulation", 'test_per_class', config.N_TEST)
    traj.f_add_parameter_to_group("simulation", 'num_epochs', config.N_EPOCHS)
    traj.f_add_parameter_to_group("simulation", 'total_per_class', config.TOTAL_SAMPLES)
    traj.f_add_parameter_to_group("simulation", 'kernel_width', config.KERNEL_W)
    traj.f_add_parameter_to_group("simulation", 'kernel_pad', config.PAD)
    traj.f_add_parameter_to_group("simulation", 'output_size', config.OUTPUT_SIZE)
    traj.f_add_parameter_to_group("simulation", 'use_gabor', config.USE_GABOR_LAYER)
    traj.f_add_parameter_to_group("simulation", 'expand', config.EXPANSION_RANGE[0])
    # traj.f_add_parameter_to_group("simulation", 'conn_dist', config.CONN_DIST)
    traj.f_add_parameter_to_group("simulation", 'prob_noise', config.PROB_NOISE_SAMPLE)
    traj.f_add_parameter_to_group("simulation", 'noisy_spikes_path', paths.root_dir_path)


    # db_path = '/home/gp283/brainscales-recognition/codebase/images_to_spikes/omniglot/spikes'
    db_path = '/home/gp283/brainscales-recognition/codebase/images_to_spikes/omniglot/spikes_shrink_%d'%config.INPUT_SHAPE[0]
    traj.f_add_parameter_to_group("simulation", 'spikes_path', db_path)

    # dbs = [ name for name in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, name)) ]
    # print(dbs)
    # dbs = [
    #     'Mkhedruli_-Georgian-', 'Tagalog', 'Ojibwe_-Canadian_Aboriginal_Syllabics-',
    #     'Asomtavruli_-Georgian-', 'Balinese', 'Japanese_-katakana-', 'Malay_-Jawi_-_Arabic-',
    #     'Armenian', 'Burmese_-Myanmar-', 'Arcadian', 'Futurama', 'Cyrillic',
    #     'Alphabet_of_the_Magi', 'Sanskrit', 'Braille', 'Bengali',
    #     'Inuktitut_-Canadian_Aboriginal_Syllabics-', 'Syriac_-Estrangelo-', 'Gujarati',
    #     'Korean', 'Early_Aramaic', 'Japanese_-hiragana-', 'Anglo-Saxon_Futhorc', 'N_Ko',
    #     'Grantha', 'Tifinagh', 'Blackfoot_-Canadian_Aboriginal_Syllabics-', 'Greek',
    #     'Hebrew', 'Latin'
    # ]

    # dbs = ['Alphabet_of_the_Magi']
    dbs = ['Futurama']
    # dbs = ['Blackfoot_-Canadian_Aboriginal_Syllabics-', 'Gujarati', 'Syriac_-Estrangelo-']

    traj.f_add_parameter_to_group("simulation", 'database', dbs)

    ## Innerloop simulator
    optimizee = OmniglotOptimizee(traj, 1234)

    # Prepare optimizee for jube runs
    JUBE_runner.prepare_optimizee(optimizee, paths.root_dir_path)

    _, dict_spec = dict_to_list(optimizee.create_individual(), get_dict_spec=True)
    step_size = np.asarray([config.ATTR_STEPS[k] for (k, spec, length) in dict_spec])
    fit_weights = [1.0, 0.1]
    if OPTIMIZER == GRADDESC:
        n_random_steps = 10
        n_iteration = 100

        parameters = RMSPropParameters(learning_rate=0.001,
                                       exploration_step_size=step_size,
                                       n_random_steps=n_random_steps,
                                       momentum_decay=0.5,
                                       n_iteration=n_iteration,
                                       stop_criterion=np.Inf,
                                       seed=99)

        optimizer = GradientDescentOptimizer(traj,
                        optimizee_create_individual=optimizee.create_individual,
                        optimizee_fitness_weights=fit_weights,
                        parameters=parameters,
                        optimizee_bounding_func=optimizee.bounding_func)

    elif OPTIMIZER == EVOSTRAT:
        optimizer_seed = 1234
        parameters = EvolutionStrategiesParameters(
            learning_rate=0.1,
            noise_std=step_size,
            mirrored_sampling_enabled=True,
            fitness_shaping_enabled=True,
            pop_size=9,
            n_iteration=1000,
            stop_criterion=np.Inf,
            seed=optimizer_seed)

        optimizer = EvolutionStrategiesOptimizer(
            traj,
            optimizee_create_individual=optimizee.create_individual,
            optimizee_fitness_weights=fit_weights,
            parameters=parameters,
            optimizee_bounding_func=optimizee.bounding_func)
    else:
        parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5,
                                                MUTPB=0.3, NGEN=100, indpb=0.02,
                                                tournsize=15, matepar=0.5,
                                                mutpar=1
                                                )

        optimizer = GeneticAlgorithmOptimizer(traj,
            optimizee_create_individual=optimizee.create_individual,
            optimizee_fitness_weights=fit_weights,
            parameters=parameters)

    # Add post processing
    ### guess this is where we want to split results from multiple runs?
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    ## Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()