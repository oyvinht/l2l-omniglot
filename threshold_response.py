import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pprint import pprint
import pynn_genn as sim
import stdp_mech as __stdp__
import neuron_model as __neuron__
backend = 'genn'
neuron_class = __neuron__.IF_curr_exp_i
# heidelberg's brainscales seems to like these params
e_rev = 92 #mV
# e_rev = 500.0 #mV

base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'v_thresh_adapt': -55.0,
    # 'e_rev_I': -e_rev, #mV
    # 'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10.,  # ms
    'tau_refrac': 0.1,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms

}

timestep = 1.0
max_w = 0.01
start_w = max_w / 2.0

tau_plus = 5.0
tau_minus = 10.0
a_plus = 0.01
a_minus = 0.005
delays = [0.1]

start_dt, num_dt = -50, 100
# start_dt, num_dt = -5, 10
sim_time = np.round(1.5 * num_dt)
start_t = sim_time - num_dt
trigger_t = start_t + (start_dt + num_dt//2)
num_neurons = num_dt

sim.setup(timestep=timestep, min_delay=timestep,
          backend='SingleThreadedCPU')


pre_spike_times = [[trigger_t + 1.0]]
trigger_spike_times = [[trigger_t, trigger_t + 1, trigger_t + 2, trigger_t + 3]]

trigger = sim.Population(1,
            sim.SpikeSourceArray(**{'spike_times': trigger_spike_times}))

post = sim.Population(1, neuron_class(**base_params))
post.record('spikes')
post.record('v_thresh_adapt')

pre = sim.Population(1,
         sim.SpikeSourceArray(**{'spike_times': pre_spike_times}))

tr2post = sim.Projection(trigger, post,
            sim.OneToOneConnector(),
            synapse_type=sim.StaticSynapse(weight=5.0, delay=0.1),
            receptor_type='excitatory', label='trigger connection')

time_dep_vars = {
    "A_plus": 0.10,
    "A_minus": 0.01,
    "mean": 0.0,
    "std": 4.0,
    "displace": 0.01,
    "maxDt": 50.0,
}

tdep = __stdp__.MyTemporalDependence(**time_dep_vars)
wdep = __stdp__.MyWeightDependence(w_min=0.0, w_max=max_w)
syn = __stdp__.MySTDPMechanism(
    timing_dependence=tdep, weight_dependence=wdep,
    weight=start_w, delay=1.0,
)
pre2post = sim.Projection(pre, post,
            sim.AllToAllConnector(), synapse_type=syn,
            receptor_type='excitatory', label='plastic connection')


sim.run(sim_time)

data = post.get_data()

sim.end()

vthresh = data.segments[0].filter(name='v_thresh_adapt')

plt.figure()
ax = plt.subplot()
plt.plot(vthresh[0])
plt.grid()
plt.show()

