import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pprint import pprint
import pynn_genn as sim
import omnigloter.stdp_mech as __stdp__
import omnigloter.neuron_model as __model__
backend = 'genn'
neuron_class = __model__.IF_curr_exp_i
# heidelberg's brainscales seems to like these params
e_rev = 92 #mV
# e_rev = 500.0 #mV

base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    # 'e_rev_I': -e_rev, #mV
    # 'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10.,  # ms
    'tau_refrac': 2.0,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms

}

timestep = 0.1
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

time_dep_vars = {
    "A_plus": 0.10,
    "A_minus": 0.000001,
    "mean": 0.0,
    "std": 1.0,
    "displace": 0.005,
    "maxDt": 80.0,
}

sim.setup(timestep=timestep, min_delay=timestep,
          backend='SingleThreadedCPU')

pprojs = {}
for delay in delays:

    a_plus_local = a_plus if delay == 1.0 else -a_plus
    a_minus_local = a_minus if delay == 1.0 else -a_minus

    projs = {}
    for dt in range(start_dt, start_dt+num_dt, 1):
        pre_spike_times = [[trigger_t + dt]]
        trigger_spike_times = [[trigger_t]]

        trigger = sim.Population(1,
                    sim.SpikeSourceArray(**{'spike_times': trigger_spike_times}))

        post = sim.Population(1, neuron_class(**base_params))
        post.record('spikes')

        pre = sim.Population(1,
                 sim.SpikeSourceArray(**{'spike_times': pre_spike_times}))

        tr2post = sim.Projection(trigger, post,
                    sim.OneToOneConnector(),
                    synapse_type=sim.StaticSynapse(weight=2.0, delay=0.1),
                    receptor_type='excitatory', label='trigger connection')


        tdep = __stdp__.MyTemporalDependence(**time_dep_vars)
        wdep = __stdp__.MyWeightDependence(w_min=0.0, w_max=max_w)
        syn = __stdp__.MySTDPMechanism(
            timing_dependence=tdep, weight_dependence=wdep,
            weight=start_w, delay=delay,
        )
        pre2post = sim.Projection(pre, post,
                    sim.AllToAllConnector(), synapse_type=syn,
                    receptor_type='excitatory', label='plastic connection')

        projs[dt] = pre2post

    pprojs[delay] = projs

sim.run(sim_time)
experiments = {}
for delay in pprojs:
    dt_dw = {}
    for dt in pprojs[delay]:
        dt_dw[dt] = (pprojs[delay][dt].getWeights(format='array')[0,0] - start_w) / max_w
    experiments[delay] = dt_dw

sim.end()



plt.figure()
ax = plt.subplot()
plt.axvline(0, linestyle='--', color='gray')
plt.axhline(0, linestyle='--', color='gray')

for delay in experiments:
    dt_dw = experiments[delay]
    dts = sorted(dt_dw.keys())
    dws = [dt_dw[dt] for dt in dts]
    plt.plot(dts, dws, label=delay)

max_dw = np.max(np.abs(dws)) * 1.5
ax.set_ylim(-max_dw, max_dw)
ax.set_xlabel(r'$\Delta t = t_{pre} - t_{post}$ [ms]')
ax.set_ylabel(r'$\Delta w $')
plt.legend()
plt.grid()
plt.show()

np.savez_compressed('delay_experiments.npz', experiments=experiments)
