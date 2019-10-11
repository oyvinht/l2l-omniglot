import numpy as np
import os

DEBUG = bool(0)
BACKEND = 'SingleThreadedCPU' if bool(0) else 'CUDA'

INF = float(10e10)

USE_GABOR_LAYER = bool(0)

SIM_NAME = 'genn'


TIMESTEP = 0.1 #ms
SAMPLE_DT = 50.0 #ms
# iw = 28
# iw = 32
# iw = 48
iw = 56
# iw = 64
# iw = 105
INPUT_SHAPE = (iw, iw)
# INPUT_DIVS = (3, 5)
# INPUT_DIVS = (3, 3)
INPUT_DIVS = (2, 2)
# INPUT_DIVS = (1, 1)
# INPUT_DIVS = (2, 3)
N_CLASSES = 2 if DEBUG else 14
N_SAMPLES = 1 if DEBUG else 17
N_EPOCHS = 1 if DEBUG else 10
N_TEST = 2 if DEBUG else 3
TOTAL_SAMPLES = N_SAMPLES * N_EPOCHS + N_TEST
DURATION = N_CLASSES * TOTAL_SAMPLES * SAMPLE_DT
PROB_NOISE_SAMPLE = 0.1


KERNEL_W = 7
N_INPUT_LAYERS = 4
PAD = KERNEL_W//2
PI_DIVS_RANGE =      (6, 7)          if DEBUG else (2, 7)
STRIDE_RANGE =       (2, 3)          if DEBUG else (1, KERNEL_W//2 + 1)
OMEGA_RANGE =        (0.5, 1.0)
EXPANSION_RANGE =    (10, 11)        if DEBUG else (10, 11)
EXP_PROB_RANGE =     (0.1, 0.11)     if DEBUG else (0.05, 0.15)
OUTPUT_PROB_RANGE =  (0.15, 0.151)   if DEBUG else (0.05, 0.15)
A_PLUS =             (0.1, 0.11)     if DEBUG else (0.01, 1.0)
A_MINUS =            (0.001, 0.0011) if DEBUG else (0.001, 0.1)
STD_DEV =            (1.0, 1.1)      if DEBUG else (0.01, 5.0)
DISPLACE =           (0.0,)#05, 0.0051) if DEBUG else (0.001, 1.0)
MAX_DT =             (80.0, 80.1)    if DEBUG else (SAMPLE_DT, SAMPLE_DT*2.0)
W_MIN_MULT =         (0.0, 0.1)      if DEBUG else (-2.0, 0.0)
W_MAX_MULT =         (1.2,)# 1.21)     if DEBUG else (0.1, 2.0)
CONN_DIST =          (15, 16)        if DEBUG else (3, 18)

GABOR_WEIGHT_RANGE = (2.0, 2.000001) if DEBUG else (1.0, 5.0)

# OUT_WEIGHT_RANGE = (0.1, 0.100000001) if DEBUG else (1.0, 5.0)
OUT_WEIGHT_RANGE = (2.0, 2.000000001) if DEBUG else (1.0, 5.0)
# OUT_WEIGHT_RANGE = (1.5, 1.500001) if DEBUG else (0.01, 0.5) ### 64x64

MUSHROOM_WEIGHT_RANGE = (0.2, 0.20000001) if DEBUG else (0.1, 1.0)
# MUSHROOM_WEIGHT_RANGE = (0.50, 0.500000001) if DEBUG else (0.05, 1.0)
# MUSHROOM_WEIGHT_RANGE = (0.025, 0.02500001) if DEBUG else (0.05, 1.0) ### for (64,64)

N_PER_CLASS = 20
OUTPUT_SIZE = N_CLASSES * N_PER_CLASS

# CONN_DIST = 3
# CONN_DIST = 9
# CONN_DIST = 15
# CONN_ANGS = 9
# CONN_RADII = [3, ]

### static weights
# gabor_weight = [1.0, 1.0, 2.0, 2.0]
# mushroom_weight = 0.25
INHIBITORY_WEIGHT = {
    'gabor': -5.0,
    'mushroom': -10.0,
    'output': -5.0,
}

EXCITATORY_WEIGHT = {
    'gabor': 3.0,
    'mushroom': 5.0,
    'output': 5.0,
}
MUSH_SELF_PROB = 0.0075

ATTRS = [
    'out_weight',
    # 'n_pi_divs', 'stride', 'omega',
     'expand', 'exp_prob', 'out_prob',
    'mushroom_weight'
]
# ATTRS += ['gabor_weight-%d'%i for i in range(N_INPUT_LAYERS)]

N_ATTRS = len(ATTRS)

ATTR2IDX = {attr: i for i, attr in enumerate(ATTRS)}

ATTR_RANGES = {
    'out_weight': OUT_WEIGHT_RANGE,
    'mushroom_weight': MUSHROOM_WEIGHT_RANGE,
    'exp_prob': EXP_PROB_RANGE,
    'out_prob': OUTPUT_PROB_RANGE,
    'conn_dist': CONN_DIST,

    'A_plus': A_PLUS,
    'A_minus': A_MINUS,
    # 'std': STD_DEV,
    # 'displace': DISPLACE,
    # 'maxDt': MAX_DT,
    'w_max_mult': W_MAX_MULT,
    'w_min_mult': W_MIN_MULT,
}

ATTR_STEPS = {
    'out_weight': 0.1,
    'mushroom_weight': 0.1,
    'exp_prob': 0.001,
    'out_prob': 0.001,
    'A_plus': 0.01,
    'A_minus': 0.01,
    'std': 0.5,
    'displace': 0.001,
    'maxDt': 10.0,
    'w_max_mult': 0.01,
    'w_min_mult': 0.01,
    'conn_dist': 5,
}

# for s in ATTRS:
#     if s.startswith('gabor_weight'):
#         ATTR_RANGES[s] = GABOR_WEIGHT_RANGE


### Neuron types
NEURON_CLASS = 'IF_curr_exp'
GABOR_CLASS = 'IF_curr_exp'
MUSHROOM_CLASS = 'IF_curr_exp_i'
INH_MUSHROOM_CLASS = 'IF_curr_exp'
OUTPUT_CLASS = 'IF_curr_exp_i'
INH_OUTPUT_CLASS = 'IF_curr_exp'

### Neuron configuration
VTHRESH = -55.0
BASE_PARAMS = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': VTHRESH,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 1.,  # ms
    'tau_syn_E': 1., # ms
    'tau_syn_I': 1., # ms
}

GABOR_PARAMS = BASE_PARAMS.copy()
MUSHROOM_PARAMS = BASE_PARAMS.copy()
MUSHROOM_PARAMS['v_thresh_adapt'] = MUSHROOM_PARAMS['v_thresh']
MUSHROOM_PARAMS['tau_thresh'] = 80.0
MUSHROOM_PARAMS['mult_thresh'] = 1.8
MUSHROOM_PARAMS['tau_syn_I'] = 5.

INH_MUSHROOM_PARAMS = BASE_PARAMS.copy()
INH_OUTPUT_PARAMS = BASE_PARAMS.copy()

OUTPUT_PARAMS = BASE_PARAMS.copy()
OUTPUT_PARAMS['v_thresh_adapt'] = OUTPUT_PARAMS['v_thresh']
OUTPUT_PARAMS['tau_thresh'] = 120.0
OUTPUT_PARAMS['mult_thresh'] = 1.8
OUTPUT_PARAMS['tau_syn_I'] = 5.




RECORD_SPIKES = [
    # 'input',
    # 'gabor',
    'mushroom',
    'inh_mushroom',
    'output',
    'inh_output',
]

RECORD_WEIGHTS = [
    # 'input to gabor',
    # 'gabor to mushroom',
    # 'input to mushroom',
    'mushroom to output'
]

# STDP_MECH = 'STDPMechanism'
#
# time_dep = 'SpikePairRule'
# time_dep_vars = dict(
#     tau_plus = 20.0,
#     tau_minus = 20.0,
#     A_plus = 0.01,
#     A_minus = 0.01,
# )
#
# weight_dep = 'AdditiveWeightDependence'
# weight_dep_vars = dict(
# )
# w_min_mult = 0.0
# w_max_mult = 1.2

STDP_MECH = 'MySTDPMechanism'

TIME_DEP = 'MyTemporalDependence'
TIME_DEP_VARS = {
    "A_plus": 0.10,
    "A_minus": 0.01,
    "mean": 0.0,
    "std": 1.0,
    "displace": 0.0,
    "maxDt": 80.0,
}

WEIGHT_DEP = 'MyWeightDependence'
WEIGHT_DEP_VARS = dict(
)
W_MIN_MULT = 0#-2.0
W_MAX_MULT = 1.2

